import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:opencv_dart/opencv_dart.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:flutter/services.dart'; // For MethodChannel
import 'package:open_file/open_file.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:video_player/video_player.dart';

import 'maskinator.dart';
import 'scanner_utils.dart';

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  DartPluginRegistrant.ensureInitialized();

  service.on('run').listen((event) async {
    print('background started');
    final List<String> framePaths = (event!['frames'] as List).cast<String>();
    final frames = framePaths.map((path) => File(path)).toList();
    
    Maskinator fastSam = Maskinator();
    List<Map<String, dynamic>> modelOut = await fastSam.runBatchIso(frames);

    final tempDir = await getTemporaryDirectory();
    final tempFile = File('${tempDir.path}/result.json');
    await tempFile.writeAsString(jsonEncode(modelOut));

    print('background ended');
    service.invoke('update', {'result_path': tempFile.path});
  });

  service.on('stopService').listen((event) {
    service.stopSelf();
  });
}

Future<void> initializeService() async {
  DartPluginRegistrant.ensureInitialized();
  final service = FlutterBackgroundService();

  // 2. Configure the service for Android and iOS.
  await service.configure(
    androidConfiguration: AndroidConfiguration(
      // This will be executed when the service is started.
      onStart: onStart,
      // To run in the foreground, we need a notification.
      isForegroundMode: true,
      autoStart: false,
    ),
    iosConfiguration: IosConfiguration(
      // This will be executed when the app is in the foreground.
      onForeground: onStart,
      autoStart: false,
    ),
  );
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializeService();
  runApp(const MaterialApp(home: MyApp()));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool devMode = false; // Toggle for development mode
  File? _videoFile;
  bool _extracting = false;
  double _frameRate = 3.0;
  bool isNotebook = true;
  Uint8List _pdfBytes = Uint8List(0);
  // Store history of extractions: each entry is {log, frames, framesWithMasks, croppedMasks}
  String? _processingMessage;
  final Map<String, dynamic> _devLog = {'log': '', 'frames': [], 'framesWithMasks': [], 'croppedMasks': []};
  void _appendLog(String msg) {
    setState(() {
      _devLog['log'] += "$msg\n";
    });
  }

  void _restartApp() {
    final service = FlutterBackgroundService();
    service.invoke('stop');
    setState(() {
      _videoFile = null;
      _extracting = false;
      _devLog['log'] = '';
      _devLog['frames'] = [];
      _devLog['frames'] = [];
      _devLog['framesWithMasks'] = [];
      _devLog['croppedMasks'] = [];
      _pdfBytes = Uint8List(0);
    });
  }

  Future<void> _pickVideo(ImageSource source) async {
    setState(() => _extracting = true);
    final picker = ImagePicker();

    final picked = await picker.pickVideo(source: source);
    if (picked != null) {
      // If the video is from the camera, rotate to horizontal if needed
      File videoFile = File(picked.path);

			setState(() {
					_devLog['log'] = '';
					_devLog['frames'] = [];
					_devLog['framesWithMasks'] = [];
					_devLog['croppedMasks'] = [];
					_videoFile = videoFile;
					_pdfBytes = Uint8List(0);
					});
			_appendLog('Selected video: ${videoFile.path}');

      processVideo();
    }
  }

  Future<void> processVideo() async {
    final List<File> frames = await _extractFramesIso(_videoFile);
    final framePaths = frames.map((f) => f.path).toList();

		setState(() {
				_processingMessage = 'You may use other apps while your video is processed';
				});

    final service = FlutterBackgroundService();
    service.startService();
    service.invoke('run', {'frames': framePaths});

    var stopwatch = Stopwatch()..start();

    service.on('update').listen((event) async {
      final String resultPath = event!['result_path'];
      final file = File(resultPath);
      final result = jsonDecode(await file.readAsString());
      await file.delete();

      final modelOut = (result as List).cast<Map<String, dynamic>>();

      var log = 'Segmented ${frames.length} frames.\n';
      for (var i = 0; i < modelOut.length; i++) {
        log += 'Frame ${i + 1}: ${modelOut[i]['stats']}\n';
      }

      List<Map<String, dynamic>> filteredOut = ScannerUtils().filter(modelOut, isNotebook, 0.20, 0.70, 0.15, 5, 1, 40);
      List<Map<String, dynamic>> finalPages = ScannerUtils().finalFrames(filteredOut);

      for (var i = 0; i < filteredOut.length; i++) {
        log += 'Frame ${i + 1}: ${filteredOut[i]['ann'].length} pages found, ${finalPages[i]['ann'].length} final doc\n';
      }

      List<img.Image> framesWithMasks = displayFramesWithMasks(frames, filteredOut);

      // Crop frames to each mask's bounding box
      List<img.Image> croppedMasks = cropFramesToMasks(frames, finalPages);

      stopwatch.stop();
      log += 'Total scanning time: ${stopwatch.elapsedMilliseconds} ms\n';

      createPdfFromImages(croppedMasks).then((pdfBytes) {
        setState(() {
          _devLog['log'] = '${_devLog['log']}$log';
          _devLog['framesWithMasks'] = framesWithMasks;
          _devLog['croppedMasks'] = croppedMasks;
          _extracting = false;
          _pdfBytes = pdfBytes;
          _processingMessage = null;
        });
      });

      service.invoke('stop');
    });
  }

  List<img.Image> displayFramesWithMasks(List<File> frames, List<Map<String, dynamic>> modelOut) {
    List<img.Image> images = [];

    for (var i = 0; i < frames.length; i++) {
      var f = frames[i];
      final bytes = f.readAsBytesSync();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw FormatException('Decoded image is null');
      final resized = img.copyResize(decoded, width: 640, height: 640);
      for (var j = 0; j < modelOut[i]['ann'].length; j++) {
        var data = modelOut[i]['ann'][j];
        final mask = (data['mask'] as List).map((dynamic row) => (row as List).cast<bool>()).toList();
        // Coordinates will be used later for drawing the rectangle
        // draw mask in transparent blue on the image
        for (int y = 0; y < resized.height; y++) {
          for (int x = 0; x < resized.width; x++) {
            if (x >= 0 && x < resized.width && y >= 0 && y < resized.height) {
              if (mask[y][x]) {
                final originalColor = resized.getPixel(x, y); // Create a consistent random color per detected object
                final random = Random(data.hashCode);
                final r = random.nextInt(256);
                final g = random.nextInt(256);
                final b = random.nextInt(256);

                final blendedColor = img.ColorRgba8(
                  ((originalColor.r as int) * 0.7 + r * 0.3).round(),
                  ((originalColor.g as int) * 0.7 + g * 0.3).round(),
                  ((originalColor.b as int) * 0.7 + b * 0.3).round(),
                  50,
                );
                resized.setPixel(x, y, blendedColor);
              }
            }
          }
        }
      }
      for (var j = 0; j < modelOut[i]['ann'].length; j++) {
        var data = modelOut[i]['ann'][j];
        final xyxy = (data['xyxy'] as List).cast<int>();

        // Use the same random color for the rectangle as used for the mask
        final random = Random(data.hashCode);
        final r = random.nextInt(256);
        final g = random.nextInt(256);
        final b = random.nextInt(256);

        img.drawRect(resized, x1: xyxy[0], y1: xyxy[1], x2: xyxy[2], y2: xyxy[3], color: img.ColorRgb8(r, g, b), thickness: 2);
      }

      images.add(resized);
    }
    return images;
  }

  // Function to crop frames to mask bounding boxes
  List<img.Image> cropFramesToMasks(
    List<File> frames,
    List<Map<String, dynamic>> filteredOut, [
    int cwidth = 640,
    int cheight = 640,
  ]) {
    List<img.Image> croppedImages = [];

    for (var i = 0; i < frames.length; i++) {
      img.Image frame = img.decodeImage(frames[i].readAsBytesSync())!;

      // For each annotation/mask in the frame
      for (var j = 0; j < filteredOut[i]['ann'].length; j++) {
        var data = filteredOut[i]['ann'][j];
        final xyxy = (data['xyxy'] as List).cast<int>();

        // Calculate the dimensions of the cropped image
        int x1 = xyxy[0];
        int y1 = xyxy[1];
        int x2 = xyxy[2];
        int y2 = xyxy[3];

        // normalize coordinates to original image size
        x1 = (x1 * frame.width / cwidth).round();
        y1 = (y1 * frame.height / cheight).round();
        x2 = (x2 * frame.width / cwidth).round();
        y2 = (y2 * frame.height / cheight).round();
        if (x2 <= x1 || y2 <= y1) continue;

        // Create a cropped copy of the image
        final cropped = img.copyCrop(frame, x: x1, y: y1, width: x2 - x1, height: y2 - y1);
        croppedImages.add(cropped);
      }
    }

    return croppedImages;
  }

  Future<Duration> getVideoDuration(File file) async {
    final controller = VideoPlayerController.file(file);
    await controller.initialize(); // This loads the video metadata
    final duration = controller.value.duration;
    await controller.dispose();
    return duration;
  }

  Future<List<File>> _extractFramesIso(File? videoFile) async {
    if (videoFile == null)
      return [];
    else {
      var stopwatch = Stopwatch()..start();

      final dir = await getTemporaryDirectory();
      final dura = (await getVideoDuration(videoFile)).inMilliseconds;

      var frames = await compute(_extractFrames, {'videoFile': videoFile, 'frameRate': _frameRate, 'dir': dir, 'duration': dura});

      final elapsedMs = stopwatch.elapsedMilliseconds;
      setState(() {
        _devLog['frames'] = frames;
        _devLog['log'] = '${_devLog['log']}Extracted ${frames.length} in $elapsedMs ms\n';
      });

      return frames;
    }
  }

  static Future<List<File>> _extractFrames(Map<String, dynamic> data) async {
    File videoFile = data['videoFile'];
    double frameRate = data['frameRate'];
    var dir = data['dir'] as Directory;
    if (videoFile == null) return [];

    var filename = videoFile.path;
    final vidCap = VideoCapture.fromFile(filename!);

    final duration = data['duration'] as int;
    final durs = duration / 1000;
    var frame_interval = (vidCap.get(5) / frameRate).round();

    List<File> frames = [];

    for (var i = 0; ; i++) {
      var success, frame;
      (success, frame) = vidCap.read();
      if (!success) break;

      if (i % frame_interval == 0) {
        final fileName = '${dir.path}/frame_$i.png';
        imwrite(fileName, frame);
        frames.add(File(fileName));
      }
    }
    
    return frames;
  }

  Future<Uint8List> createPdfFromImages(List<img.Image> images) async {
    // Create a PDF document
    final pdf = pw.Document();

    for (final image in images) {
      // Convert img.Image to Uint8List in PNG format
      final pngBytes = img.encodePng(image);

      // Create a custom page format that matches the image dimensions exactly
      // Convert image dimensions from pixels to points (1 point = 1/72 inch)
      // We'll use a standard DPI of 72 for simplicity, making 1 pixel = 1 point
      final pageFormat = PdfPageFormat(
        image.width.toDouble(),
        image.height.toDouble(),
        marginAll: 0, // No margins
      );

      // Add each image as a full page with exact dimensions
      pdf.addPage(
        pw.Page(
          pageFormat: pageFormat,
          margin: pw.EdgeInsets.zero, // Ensure no margins
          build: (pw.Context context) {
            return pw.Image(
              pw.MemoryImage(pngBytes),
              fit: pw.BoxFit.fill, // Fill the entire page exactly
              width: image.width.toDouble(),
              height: image.height.toDouble(),
            );
          },
        ),
      );
    }

    // Return the PDF as bytes
    return pdf.save();
  }

  Future<void> savePdf(List<img.Image> images) async {
    if (_pdfBytes == Uint8List(0)) return;

    try {
      final Directory appStore = await getApplicationDocumentsDirectory();

      final String documentsPath = appStore.path;
      final String fileName = 'flipatron_scan_${DateTime.now().millisecondsSinceEpoch}.pdf';
      final String filePath = '$documentsPath/$fileName';

      final File file = File(filePath);
      await file.writeAsBytes(_pdfBytes);

      await File(filePath).writeAsBytes(_pdfBytes);

      await OpenFile.open(
        filePath,
        type: 'application/pdf',
        //       uti: 'com.adobe.pdf',
      );
      setState(() {
        _devLog['log'] += "PDF saved to: $filePath\n";
      });
    } catch (e) {
      print('Error saving PDF: $e');
    }
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF000000), // OLED black background
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            children: [
              // Toggle button between notebook and pages
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: !isNotebook ? const Color(0xFF4F8CFF).withOpacity(0.15) : Colors.transparent,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Pages',
                        style: TextStyle(
                          color: !isNotebook ? const Color(0xFFB3C6FF) : const Color(0xFFB3C6FF).withOpacity(0.6),
                          fontSize: 16,
                          fontWeight: !isNotebook ? FontWeight.w700 : FontWeight.w500,
                        ),
                      ),
                    ),
                    const SizedBox(width: 20),
                    GestureDetector(
                      onTap: _extracting
                          ? null
                          : () {
                              setState(() {
                                isNotebook = !isNotebook;
                                // Add animation effect for the page-to-book transition
                                if (isNotebook) {
                                  // Animating to notebook mode
                                  Future.delayed(Duration(milliseconds: 100), () {
                                    setState(() {}); // Trigger rebuild for animation
                                  });
                                } else {
                                  // Animating to page mode
                                  Future.delayed(Duration(milliseconds: 100), () {
                                    setState(() {}); // Trigger rebuild for animation
                                  });
                                }
                              });
                            },
                      child: Opacity(
                        opacity: _extracting ? 0.5 : 1.0,
                        child: Container(
                          width: 160,
                          height: 80,
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(20),
                            gradient: const LinearGradient(
                              colors: [Color(0xFF1A2233), Color(0xFF2D3546)],
                              begin: Alignment.bottomLeft,
                              end: Alignment.topRight,
                            ),
                            boxShadow: [
                              BoxShadow(color: Colors.black.withOpacity(0.2), blurRadius: 8, offset: const Offset(0, 4)),
                            ],
                          ),
                          child: AnimatedAlign(
                            duration: const Duration(milliseconds: 250),
                            alignment: isNotebook ? Alignment.centerRight : Alignment.centerLeft,
                            child: Container(
                              width: 80,
                              height: 64,
                              decoration: BoxDecoration(
                                gradient: const LinearGradient(
                                  colors: [Color(0xFF4F8CFF), Color(0xFF3A70E0)],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                                borderRadius: BorderRadius.circular(16),
                                boxShadow: [
                                  BoxShadow(
                                    color: const Color(0xFF3A70E0).withOpacity(0.4),
                                    blurRadius: 6,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: Center(
                                child: Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    // Notebook icon
                                    Transform(
                                      alignment: Alignment.center,
                                      transform: Matrix4.identity()
                                        ..setEntry(3, 2, 0.001)
                                        ..rotateY(isNotebook ? 0 : 3.14159),
                                      child: Opacity(
                                        opacity: isNotebook ? 1.0 : 0.0,
                                        child: const Icon(Icons.menu_book_rounded, color: Colors.white, size: 32),
                                      ),
                                    ),
                                    // Page icon with animation
                                    Transform(
                                      alignment: Alignment.center,
                                      transform: Matrix4.identity()
                                        ..setEntry(3, 2, 0.001)
                                        ..rotateY(isNotebook ? 3.14159 : 0),
                                      child: Opacity(
                                        opacity: isNotebook ? 0.0 : 1.0,
                                        child: const Icon(Icons.insert_drive_file_outlined, color: Colors.white, size: 32),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 20),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: isNotebook ? const Color(0xFF4F8CFF).withOpacity(0.15) : Colors.transparent,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Notebook',
                        style: TextStyle(
                          color: isNotebook ? const Color(0xFFB3C6FF) : const Color(0xFFB3C6FF).withOpacity(0.6),
                          fontSize: 16,
                          fontWeight: isNotebook ? FontWeight.w700 : FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              // Developer mode toggle
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: devMode ? const Color(0xFF4F8CFF).withOpacity(0.15) : Colors.transparent,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'Developer Mode',
                        style: TextStyle(
                          color: devMode ? const Color(0xFFB3C6FF) : const Color(0xFFB3C6FF).withOpacity(0.6),
                          fontSize: 14,
                          fontWeight: devMode ? FontWeight.w700 : FontWeight.w500,
                        ),
                      ),
                    ),
                    const SizedBox(width: 20),
                    GestureDetector(
                      onTap: _extracting
                          ? null
                          : () {
                              setState(() {
                                devMode = !devMode;
                              });
                            },
                      child: Opacity(
                        opacity: _extracting ? 0.5 : 1.0,
                        child: Container(
                          width: 120,
                          height: 50,
                          padding: const EdgeInsets.all(6),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(16),
                            gradient: const LinearGradient(
                              colors: [Color(0xFF1A2233), Color(0xFF2D3546)],
                              begin: Alignment.bottomLeft,
                              end: Alignment.topRight,
                            ),
                            boxShadow: [
                              BoxShadow(color: Colors.black.withOpacity(0.2), blurRadius: 6, offset: const Offset(0, 2)),
                            ],
                          ),
                          child: AnimatedAlign(
                            duration: const Duration(milliseconds: 250),
                            alignment: devMode ? Alignment.centerRight : Alignment.centerLeft,
                            child: Container(
                              width: 50,
                              height: 38,
                              decoration: BoxDecoration(
                                gradient: const LinearGradient(
                                  colors: [Color(0xFF4F8CFF), Color(0xFF3A70E0)],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                                borderRadius: BorderRadius.circular(12),
                                boxShadow: [
                                  BoxShadow(
                                    color: const Color(0xFF3A70E0).withOpacity(0.4),
                                    blurRadius: 4,
                                    offset: const Offset(0, 1),
                                  ),
                                ],
                              ),
                              child: Center(
                                child: Icon(devMode ? Icons.code : Icons.cancel_rounded, color: Colors.white, size: 20),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              // FPS slider
              if (devMode)
                Row(
                  children: [
                    Expanded(
                      child: Row(
                        children: [
                          const Text(
                            'FPS:',
                            style: TextStyle(
                              color: Color(0xFFB3C6FF), // light blue for OLED
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Slider(
                              value: _frameRate,
                              min: 1.0,
                              max: 30.0,
                              divisions: 29,
                              label: _frameRate.round().toString(),
                              onChanged: _extracting
                                  ? null
                                  : (val) {
                                      setState(() {
                                        _frameRate = val.roundToDouble();
                                      });
                                    },
                              activeColor: _extracting ? Color(0xFF222B3A) : Color(0xFF4F8CFF),
                              inactiveColor: Color(0xFF222B3A),
                              thumbColor: _extracting ? Color(0xFF666666) : Colors.white,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(color: Color(0xFF1A2233), borderRadius: BorderRadius.circular(8)),
                            child: Text(
                              _frameRate.round().toString(),
                              style: const TextStyle(color: Color(0xFF4F8CFF), fontWeight: FontWeight.bold, fontSize: 16),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                height: 70,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    gradient: _extracting
                        ? const LinearGradient(
                            colors: [Color(0xFFFF4F4F), Color(0xFF331A1A)],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          )
                        : const LinearGradient(
                            colors: [Color(0xFF4F8CFF), Color(0xFF1A2233)],
                            begin: Alignment.centerLeft,
                            end: Alignment.centerRight,
                          ),
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: ElevatedButton.icon(
                    icon: _extracting
                        ? const Icon(Icons.cancel_rounded, size: 36, color: Colors.white)
                        : const Icon(Icons.scanner_rounded, size: 36, color: Colors.white),
                    label: Text(
                      _extracting ? 'Cancel' : 'Select Video',
                      style: const TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.white),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                    onPressed: _extracting
                        ? () {
                            _restartApp();
                          }
                        : () async {
                            showModalBottomSheet(
                              context: context,
                              backgroundColor: const Color(0xFF10131A),
                              shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
                              builder: (context) {
                                return Padding(
                                  padding: const EdgeInsets.all(24.0),
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      SizedBox(
                                        width: double.infinity,
                                        height: 60,
                                        child: DecoratedBox(
                                          decoration: BoxDecoration(
                                            gradient: const LinearGradient(
                                              colors: [Color(0xFF4F8CFF), Color(0xFF1A2233)],
                                              begin: Alignment.centerLeft,
                                              end: Alignment.centerRight,
                                            ),
                                            borderRadius: BorderRadius.circular(14),
                                          ),
                                          child: ElevatedButton.icon(
                                            icon: const Icon(Icons.videocam_rounded, size: 28, color: Colors.white),
                                            label: const Text(
                                              'Record Video',
                                              style: TextStyle(fontSize: 22, color: Colors.white),
                                            ),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: Colors.transparent,
                                              shadowColor: Colors.transparent,
                                              foregroundColor: Colors.white,
                                              elevation: 0,
                                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                                            ),
                                            onPressed: () {
                                              Navigator.pop(context);
                                              _pickVideo(ImageSource.camera);
                                            },
                                          ),
                                        ),
                                      ),
                                      const SizedBox(height: 16),
                                      SizedBox(
                                        width: double.infinity,
                                        height: 60,
                                        child: DecoratedBox(
                                          decoration: BoxDecoration(
                                            gradient: const LinearGradient(
                                              colors: [Color(0xFF4F8CFF), Color(0xFF1A2233)],
                                              begin: Alignment.centerLeft,
                                              end: Alignment.centerRight,
                                            ),
                                            borderRadius: BorderRadius.circular(14),
                                          ),
                                          child: ElevatedButton.icon(
                                            icon: const Icon(Icons.video_file_rounded, size: 28, color: Colors.white),
                                            label: const Text(
                                              'Pick from Gallery',
                                              style: TextStyle(fontSize: 22, color: Colors.white),
                                            ),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: Colors.transparent,
                                              shadowColor: Colors.transparent,
                                              foregroundColor: Colors.white,
                                              elevation: 0,
                                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                                            ),
                                            onPressed: () {
                                              Navigator.pop(context);
                                              _pickVideo(ImageSource.gallery);
                                            },
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                );
                              },
                            );
                          },
                  ),
                ),
              ),
              if (_extracting)
                const Padding(
                  padding: EdgeInsets.all(6.0),
                  child: LinearProgressIndicator(color: Color(0xFF4F8CFF), backgroundColor: Color(0xFF222B3A)),
                ),
              if (_processingMessage != null)
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    _processingMessage!,
                    style: const TextStyle(color: Color(0xFFB3C6FF), fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                ),
              const SizedBox(height: 16), // Show extraction history
              if (_devLog['frames'] != null && (_devLog['frames'] as List).isNotEmpty)
                Expanded(
                  child: ListView(
                    children: [
                      Container(
                        margin: const EdgeInsets.only(bottom: 24),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF10131A),
                          borderRadius: BorderRadius.circular(18),
                          boxShadow: [
                            BoxShadow(color: Colors.blueGrey.withOpacity(0.08), blurRadius: 12, offset: const Offset(0, 4)),
                          ],
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            if (devMode && (_devLog['log'] as String).isNotEmpty)
                              Text(
                                _devLog['log'],
                                style: const TextStyle(fontFamily: 'monospace', color: Color(0xFFB3C6FF), fontSize: 13),
                              ),
                            if ((_devLog['frames'] as List).isNotEmpty)
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  if (devMode)
                                    const Text(
                                      "Original Frames:",
                                      style: TextStyle(color: Color(0xFFB3C6FF), fontSize: 16, fontWeight: FontWeight.bold),
                                    ),
                                  if (devMode)
                                    SizedBox(
                                      height: 180,
                                      child: ListView.builder(
                                        scrollDirection: Axis.horizontal,
                                        itemCount: (_devLog['frames'] as List).length,
                                        itemBuilder: (context, fidx) {
                                          final frames = _devLog['frames'] as List<File>;
                                          return Padding(
                                            padding: const EdgeInsets.all(8.0),
                                            child: ClipRRect(
                                              borderRadius: BorderRadius.circular(6),
                                              child: Image.file(frames[fidx]),
                                            ),
                                          );
                                        },
                                      ),
                                    ), // Display frames with masks
                                  if (devMode && (_devLog['framesWithMasks'] as List).isNotEmpty)
                                    Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        const Padding(
                                          padding: EdgeInsets.symmetric(vertical: 12.0),
                                          child: Text(
                                            "Frames with Masks:",
                                            style: TextStyle(color: Color(0xFFB3C6FF), fontSize: 16, fontWeight: FontWeight.bold),
                                          ),
                                        ),
                                        for (var i in _devLog['framesWithMasks'] as List<img.Image>)
                                          Padding(
                                            padding: const EdgeInsets.only(bottom: 16.0),
                                            child: ClipRRect(
                                              borderRadius: BorderRadius.circular(16),
                                              child: Image.memory(img.encodeJpg(i), fit: BoxFit.cover),
                                            ),
                                          ),
                                      ],
                                    ),
                                  // Display cropped masks
                                  if ((_devLog['croppedMasks'] as List?)?.isNotEmpty ?? false)
                                    Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        const Padding(
                                          padding: EdgeInsets.symmetric(vertical: 12.0),
                                          child: Text(
                                            "Scan Sucessful!",
                                            style: TextStyle(color: Color(0xFFB3C6FF), fontSize: 16, fontWeight: FontWeight.bold),
                                          ),
                                        ),
                                        for (var i in _devLog['croppedMasks'] as List<img.Image>)
                                          Padding(
                                            padding: const EdgeInsets.only(bottom: 16.0),
                                            child: ClipRRect(
                                              borderRadius: BorderRadius.circular(16),
                                              child: Image.memory(img.encodeJpg(i), fit: BoxFit.cover),
                                            ),
                                          ),
                                      ],
                                    ),
                                ],
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              // Save PDF button locked to bottom
              if ((_devLog['croppedMasks'] as List?)?.isNotEmpty ?? false)
                Padding(
                  padding: const EdgeInsets.only(top: 16.0),
                  child: SizedBox(
                    width: double.infinity,
                    height: 60,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF4CAF50), Color(0xFF2E7D32)],
                          begin: Alignment.centerLeft,
                          end: Alignment.centerRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(color: const Color(0xFF4CAF50).withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 4)),
                        ],
                      ),
                      child: ElevatedButton.icon(
                        icon: const Icon(Icons.picture_as_pdf, size: 28, color: Colors.white),
                        label: const Text(
                          'Save PDF',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          foregroundColor: Colors.white,
                          elevation: 0,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                        ),
                        onPressed: () async {
                          final croppedMasks = _devLog['croppedMasks'] as List<img.Image>;
                          savePdf(croppedMasks);
                        },
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

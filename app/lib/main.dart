import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:opencv_dart/opencv_dart.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:flutter/services.dart'; // For MethodChannel
import 'package:open_file/open_file.dart';
import 'package:video_player/video_player.dart';
//import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';

import 'maskinator.dart';
import 'scanner_utils.dart';

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  DartPluginRegistrant.ensureInitialized();

  service.on('run').listen((event) async {
    try {
      print('SERVICE background started');
      final bool isNotebook = (event!['isNotebook'] as bool);
      final List<String> framePaths = (event!['frames'] as List).cast<String>();
      final frames = framePaths.map((path) => File(path)).toList();
      print('SERVICE frames extracted from paths');

      Maskinator fastSam = Maskinator();
      List<Map<String, dynamic>> modelOut = await fastSam.runBatchIso(frames);
      print('finished runBatchIso');

      // Filtering and final frame selection inside background process
      List<Map<String, dynamic>> filteredOut = ScannerUtils().filter(modelOut, isNotebook, 0.20, 0.70, 0.15, 5, 1, 40);
      List<Map<String, dynamic>> finalPages = ScannerUtils().finalFrames(filteredOut);

      // Only keep bounding boxes for cropping in the main process
      List<Map<String, dynamic>> filteredOutBboxes = filteredOut
          .map(
            (frame) => {
              'ann': (frame['ann'] as List).map((ann) => {'xyxy': ann['xyxy']}).toList(),
            },
          )
          .toList();
      List<Map<String, dynamic>> finalPagesBboxes = finalPages
          .map(
            (frame) => {
              'ann': (frame['ann'] as List).map((ann) => {'xyxy': ann['xyxy']}).toList(),
            },
          )
          .toList();

      final tempDir = await getTemporaryDirectory();
      print('Temporary dir for background servicee $tempDir');
      final tempFile = File('${tempDir.path}/result.json');
      await tempFile.writeAsString(jsonEncode({'filteredOut': filteredOutBboxes, 'finalPages': finalPagesBboxes}));

      print('SERVICE background ended');
      service.invoke('update', {'result_path': tempFile.path});
      service.stopSelf();
    } catch (e) {
      service.invoke('error', {'error': e.toString()});
      service.stopSelf();
    }
  });

  service.on('stopService').listen((event) {
    print('SERVICE stopped');
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

Future<void> requestNotificationPermission() async {
  PermissionStatus status = await Permission.notification.status;

  if (status.isDenied) {
    // Permission is denied, request it
    status = await Permission.notification.request();
  }

  if (status.isGranted) {
    // Permission granted, proceed with notifications
    print('Notification permission granted');
  } else if (status.isPermanentlyDenied) {
    // Permission permanently denied, open app settings
    openAppSettings();
  } else {
    // Permission denied or unknown, handle accordingly
    print('Notification permission status: $status');
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializeService();
  requestNotificationPermission();
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

  static Future<dynamic> _loadAndDecodeJson(String resultPath) async {
    final file = File(resultPath);
    final result = jsonDecode(await file.readAsString());
    await file.delete();
    return result;
  }

  void _restartApp() {
    final service = FlutterBackgroundService();
    service.invoke('stopService');
    setState(() {
      _videoFile = null;
      _extracting = false;
      _devLog['log'] = '';
      _devLog['frames'] = [];
      _devLog['framesWithMasks'] = [];
      _devLog['croppedMasks'] = [];
      _pdfBytes = Uint8List(0);
      _processingMessage = '';
    });
  }

  Future<void> _pickVideo(ImageSource source) async {
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
    print('IN processVideo');

    setState(() => _extracting = true);
    final List<File> frames = await _extractFramesIso(_videoFile);
    final framePaths = frames.map((f) => f.path).toList();

    setState(() {
      _processingMessage = 'No need to stick around!';
    });

    // Check if the background service is already running
    final service = FlutterBackgroundService();
    final isRunning = await service.isRunning();
    print('IS RUNNING: ${isRunning ? 'running' : 'stopped'}');
    if(isRunning) service.invoke("stopService");
    final isRunning2 = await service.isRunning();
    print('IS RUNNING: ${isRunning2 ? 'running' : 'stopped'}');
    await service.startService();
    final isRunning3 = await service.isRunning();
    print('IS RUNNING: ${isRunning3 ? 'running' : 'stopped'}');
    service.invoke('run', {'frames': framePaths, 'isNotebook': isNotebook});
    print('after invoke run');

    var stopwatch = Stopwatch()..start();

    service.on('update').listen((event) async {
      print('SERVICE update recieved');
      final String resultPath = event!['result_path'];
      final result = await compute(_loadAndDecodeJson, resultPath);
      final filteredOut = (result['filteredOut'] as List).cast<Map<String, dynamic>>();
      final finalPages = (result['finalPages'] as List).cast<Map<String, dynamic>>();

      var log = 'Segmented ${frames.length} frames.\n';
      for (var i = 0; i < filteredOut.length; i++) {
        log += 'Frame ${i + 1}: ${filteredOut[i]['ann'].length} pages found, ${finalPages[i]['ann'].length} final doc\n';
      }

      // Crop frames to each mask's bounding box
      List<img.Image> croppedMasks = cropFramesToMasks(frames, finalPages);
      // Display frames with bounding boxes only
      List<img.Image> framesWithMasks = displayFramesWithMasks(frames, filteredOut);

      stopwatch.stop();
      log += 'Total scanning time: ${stopwatch.elapsedMilliseconds} ms\n';

      createPdfFromImages(croppedMasks).then((pdfBytes) {
        setState(() {
          _devLog['log'] = '${_devLog['log']}${log}';
          _devLog['framesWithMasks'] = framesWithMasks;
          _devLog['croppedMasks'] = croppedMasks;
          _extracting = false;
          _pdfBytes = pdfBytes;
          _processingMessage = null;
        });
      });

      service.invoke('stop');
    });

    service.on('error').listen((event) {
      _restartApp();
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Error'),
          content: Text(event!['error']),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
    });
  }

  List<img.Image> displayFramesWithMasks(List<File> frames, List<Map<String, dynamic>> bboxOut) {
    List<img.Image> images = [];

    for (var i = 0; i < frames.length; i++) {
      var f = frames[i];
      final bytes = f.readAsBytesSync();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw FormatException('Decoded image is null');
      final resized = img.copyResize(decoded, width: 640, height: 640);

      // Draw only bounding boxes from bboxOut
      for (var j = 0; j < bboxOut[i]['ann'].length; j++) {
        var data = bboxOut[i]['ann'][j];
        final xyxy = (data['xyxy'] as List).cast<int>();

        // Use a random color for each bounding box
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

    for (var i = 0; i < frames.length && i < filteredOut.length; i++) {
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
      final duration = (await getVideoDuration(videoFile)).inMilliseconds;

      var frames = await compute(_extractFrames, {
        'videoFile': videoFile,
        'frameRate': _frameRate,
        'dir': dir,
        'duration': duration,
      });

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

    var frames = <File>[];

    final vidCap = VideoCapture.fromFile(videoFile.path);

    final duration = data['duration'] as int;
    var frame_interval = (vidCap.get(5) / frameRate).round();

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

      await OpenFile.open(filePath, type: 'application/pdf');
      setState(() {
        _devLog['log'] += "PDF saved to: $filePath\n";
      });
    } catch (e) {
      print('Error saving PDF: $e');
    }
  }

  Widget build(BuildContext context) {
    SystemChrome.setSystemUIOverlayStyle(
      const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light, // Android
        statusBarBrightness: Brightness.light, // iOS
      ),
    );
    return Scaffold(
      backgroundColor: const Color(0xFF000000), // OLED black background
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.only(bottom: 0.0, left: 5.0, right: 5.0),
          child: Stack(
            children: [
              Column(
                children: [
                  // Toggle button between notebook and pages
                  Padding(
                    padding: const EdgeInsets.only(bottom: 16.0),
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
                                      Future.delayed(const Duration(milliseconds: 70), () {
                                        setState(() {}); // Trigger rebuild for animation
                                      });
                                    } else {
                                      // Animating to page mode
                                      Future.delayed(const Duration(milliseconds: 70), () {
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
                                duration: const Duration(milliseconds: 200),
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
                              const SizedBox(width: 10),
                              const Text(
                                'FPS:',
                                style: TextStyle(
                                  color: Color(0xFFB3C6FF), // light blue for OLED
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
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
                                  activeColor: _extracting ? const Color(0xFF222B3A) : const Color(0xFF4F8CFF),
                                  inactiveColor: const Color(0xFF222B3A),
                                  thumbColor: _extracting ? const Color(0xFF666666) : Colors.white,
                                ),
                              ),
                              const SizedBox(width: 10),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                decoration: BoxDecoration(color: const Color(0xFF1A2233), borderRadius: BorderRadius.circular(8)),
                                child: Text(
                                  _frameRate.round().toString(),
                                  style: const TextStyle(color: Color(0xFF4F8CFF), fontWeight: FontWeight.bold, fontSize: 16),
                                ),
                              ),
                              const SizedBox(width: 10),
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
                                  shape: const RoundedRectangleBorder(
                                    borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                                  ),
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
                                                style: TextStyle(
                                                  color: Color(0xFFB3C6FF),
                                                  fontSize: 16,
                                                  fontWeight: FontWeight.bold,
                                                ),
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
                                                style: TextStyle(
                                                  color: Color(0xFFB3C6FF),
                                                  fontSize: 16,
                                                  fontWeight: FontWeight.bold,
                                                ),
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
                ],
              ),
              // Save PDF button locked to bottom
              if (!_extracting && (_devLog['croppedMasks'] as List).isEmpty && (_devLog['frames'] as List).isEmpty == false)
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 10.0, left: 10.0, right: 10.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 24.0),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFFFF4F4F), Color(0xFFD32F2F)],
                          begin: Alignment.centerLeft,
                          end: Alignment.centerRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(color: const Color(0xFFD32F2F).withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 4)),
                        ],
                      ),
                      child: const Text(
                        'No pages detected, you may have used incorrect settings',
                        style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                )
              else if ((_devLog['croppedMasks'] as List?)?.isNotEmpty ?? false)
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 5.0, left: 10.0, right: 10.0),
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
                ),
            ],
          ),
        ),
      ),
    );
  }
}

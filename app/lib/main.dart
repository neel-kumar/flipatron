import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MaterialApp(home: MyApp()));
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File? _videoFile;
  bool _extracting = false;
  String _log = '';
  final TextEditingController _frameRateController = TextEditingController(
    text: '1',
  );
  double _frameRate = 1.0;

  // Store history of extractions: each entry is {log, frames}
  final List<Map<String, dynamic>> _extractionHistory = [];
  img.Image? _lastResizedImage;

  void _appendLog(String msg) {
    setState(() {
      _log += msg + '\n';
    });
  }

  Future<void> _pickVideo(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: source);
    if (picked != null) {
      // If the video is from the camera, rotate to horizontal if needed
      File videoFile = File(picked.path);
      if (source == ImageSource.camera) {
        final tempDir = await getTemporaryDirectory();
        final rotatedPath =
            '${tempDir.path}/rotated_${DateTime.now().millisecondsSinceEpoch}.mp4';
        // Use ffmpeg to rotate the video to horizontal (landscape)
        // This rotates 90 degrees clockwise if the video is portrait
        // and does nothing if already landscape
        // The transpose=1 filter rotates 90 degrees clockwise
        // The select filter only applies if height > width
        final probeCmd =
            '-v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "${videoFile.path}"';
        final probeResult = await FFmpegKit.executeWithArguments(
          probeCmd.split(' '),
        );
        final probeOutput = await probeResult.getOutput();
        int width = 0, height = 0;
        if (probeOutput != null && probeOutput.contains('x')) {
          final parts = probeOutput.trim().split('x');
          if (parts.length == 2) {
            width = int.tryParse(parts[0]) ?? 0;
            height = int.tryParse(parts[1]) ?? 0;
          }
        }
        if (height > width) {
          // Portrait, rotate to landscape
          final rotateCmd =
              '-i "${videoFile.path}" -vf "transpose=1" -c:a copy "$rotatedPath" -y';
          await FFmpegKit.execute(rotateCmd);
          videoFile = File(rotatedPath);
        }
      }

      // Delete old saved frames, needs to me implemented

      setState(() {
        _videoFile = videoFile;
        _log = '';
      });
      _appendLog('Selected video: \\${videoFile.path}');

      await _extractFrames();
    }
  }

  Future<void> _extractFrames() async {
    if (_videoFile == null) return;
    setState(() => _extracting = true);
    final dir = await getTemporaryDirectory();
    final framesDir = Directory('${dir.path}/frames');
    if (framesDir.existsSync()) {
      try {
        framesDir.deleteSync(recursive: true);
      } catch (_) {}
    }
    framesDir.createSync();
    final videoBase = _videoFile!.path
        .split(Platform.pathSeparator)
        .last
        .split('.')
        .first;
    final outPattern = '${framesDir.path}/' + videoBase + '_%05d.jpg';
    final cmd = '-i "${_videoFile!.path}" -vf fps=$_frameRate "$outPattern"';
    await FFmpegKit.execute(cmd);
    final RegExp framePattern = RegExp(
      '^' + RegExp.escape(videoBase) + r'_\d{5}\.jpg',
    );
    final files =
        framesDir
            .listSync()
            .whereType<File>()
            .where(
              (f) => framePattern.hasMatch(
                f.path.split(Platform.pathSeparator).last,
              ),
            )
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    setState(() {
      _extracting = false;
      // Save this extraction to history
      _extractionHistory.insert(0, {
        'log': _log + 'Extracted ${files.length} frames.\n',
        'frames': List<File>.from(files),
      });
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      setState(() {});
    });
    _appendLog('Extracted ${files.length} frames.');

    _segmentFrames(files);
  }

  Future<void> _segmentFrames(List<File> frames) async {
    final gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
    final delegate = switch (Platform.operatingSystem) {
      "ios" => GpuDelegate(),
      "android" => gpuDelegateV2,
      _ => XNNPackDelegate(),
    };

    final interpreter = await Interpreter.fromAsset(
      'assets/FastSam-X.tflite',
      options: InterpreterOptions()..addDelegate(delegate),
    );
    final inputTensors = interpreter.getInputTensors();
    print('BAJA Input tensors: ${inputTensors.toString()}');

    final outputTensors = interpreter.getOutputTensors();
    print('BAJA Ouput tensors 0: ${outputTensors[0].shape.toString()}');
    print('BAJA Ouput tensors 1: ${outputTensors[1].shape.toString()}');

    final output = {
      0: List<List<List<num>>>.filled(
        1,
        List<List<num>>.filled(
          outputTensors[0].shape[1],
          List<num>.filled(outputTensors[0].shape[2], 0),
        ),
      ),
      1: List<List<List<List<num>>>>.filled(
        1,
        List<List<List<num>>>.filled(
          outputTensors[1].shape[1],
          List<List<num>>.filled(
            outputTensors[1].shape[2],
            List<num>.filled(outputTensors[1].shape[3], 0),
          ),
        ),
      ),
    };
    print('BAJA Ouput[0] matrix shape: ${output[0]!.shape.toString()}');
    print('BAJA Output[1] matrix shape: ${output[1]!.shape.toString()}');

    final stopwatch = Stopwatch()..start();
    for (var frame in frames) {
      final input = await _filePreprocess(frame);
      interpreter.runForMultipleInputs([
        [input],
      ], output);
    }

    stopwatch.stop();
    final elapsedMs = stopwatch.elapsedMilliseconds;
    print('BAJA Output interpreter: ${output.toString()}');
    print('Segmentation inference time: ${elapsedMs} ms');

    // Add inference time to the extraction log for the current video
    setState(() {
      if (_extractionHistory.isNotEmpty) {
        final last = _extractionHistory.first;
        last['log'] =
            '${last['log'] as String}Segmentation inference time: ${elapsedMs} ms\nAverage Segmentation time: ${(elapsedMs / frames.length).round()} ms';
      }
    });
  }

  Future<List<List<Object>>?> _filePreprocess(File image) async {
    // Read image from file
    final bytes = await image.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw FormatException('Decoded image is null');
    final resized = img.copyResize(decoded, width: 640, height: 640);
    final imageMatrix = List.generate(
      resized.height,
      (y) => List.generate(resized.width, (x) {
        final pixel = resized.getPixel(x, y);
        return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
      }),
    );
    return imageMatrix;
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF000000), // OLED black background
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            children: [
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
                            onChanged: (val) {
                              setState(() {
                                _frameRate = val.roundToDouble();
                                _frameRateController.text = _frameRate
                                    .toString();
                              });
                            },
                            activeColor: Color(0xFF4F8CFF),
                            inactiveColor: Color(0xFF222B3A),
                            thumbColor: Colors.white,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: Color(0xFF1A2233),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            _frameRate.round().toString(),
                            style: const TextStyle(
                              color: Color(0xFF4F8CFF),
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 30),
              SizedBox(
                width: double.infinity,
                height: 70,
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF4F8CFF), Color(0xFF1A2233)],
                      begin: Alignment.centerLeft,
                      end: Alignment.centerRight,
                    ),
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: ElevatedButton.icon(
                    icon: const Icon(
                      Icons.scanner_rounded,
                      size: 36,
                      color: Colors.white,
                    ),
                    label: const Text(
                      'Select Video',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                    onPressed: () async {
                      showModalBottomSheet(
                        context: context,
                        backgroundColor: const Color(0xFF10131A),
                        shape: const RoundedRectangleBorder(
                          borderRadius: BorderRadius.vertical(
                            top: Radius.circular(24),
                          ),
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
                                        colors: [
                                          Color(0xFF4F8CFF),
                                          Color(0xFF1A2233),
                                        ],
                                        begin: Alignment.centerLeft,
                                        end: Alignment.centerRight,
                                      ),
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    child: ElevatedButton.icon(
                                      icon: const Icon(
                                        Icons.videocam_rounded,
                                        size: 28,
                                        color: Colors.white,
                                      ),
                                      label: const Text(
                                        'Record Video',
                                        style: TextStyle(
                                          fontSize: 22,
                                          color: Colors.white,
                                        ),
                                      ),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.transparent,
                                        shadowColor: Colors.transparent,
                                        foregroundColor: Colors.white,
                                        elevation: 0,
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(
                                            14,
                                          ),
                                        ),
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
                                        colors: [
                                          Color(0xFF4F8CFF),
                                          Color(0xFF1A2233),
                                        ],
                                        begin: Alignment.centerLeft,
                                        end: Alignment.centerRight,
                                      ),
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    child: ElevatedButton.icon(
                                      icon: const Icon(
                                        Icons.video_file_rounded,
                                        size: 28,
                                        color: Colors.white,
                                      ),
                                      label: const Text(
                                        'Pick from Gallery',
                                        style: TextStyle(
                                          fontSize: 22,
                                          color: Colors.white,
                                        ),
                                      ),
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: Colors.transparent,
                                        shadowColor: Colors.transparent,
                                        foregroundColor: Colors.white,
                                        elevation: 0,
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(
                                            14,
                                          ),
                                        ),
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
                  padding: EdgeInsets.all(24.0),
                  child: CircularProgressIndicator(color: Color(0xFF4F8CFF)),
                ),
              const SizedBox(height: 16),
              // Show extraction history
              Expanded(
                child: Column(
                  children: [
                    if (_lastResizedImage != null)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 16.0),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.memory(
                            img.encodeJpg(_lastResizedImage!),
                            width: 220,
                            height: 220,
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _extractionHistory.length,
                        itemBuilder: (context, idx) {
                          final entry = _extractionHistory[idx];
                          final log = entry['log'] as String;
                          final frames = entry['frames'] as List<File>;
                          return Container(
                            margin: const EdgeInsets.only(bottom: 24),
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: const Color(0xFF10131A),
                              borderRadius: BorderRadius.circular(18),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.blueGrey.withOpacity(0.08),
                                  blurRadius: 12,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  log,
                                  style: const TextStyle(
                                    fontFamily: 'monospace',
                                    color: Color(0xFFB3C6FF),
                                    fontSize: 16,
                                  ),
                                ),
                                if (frames.isNotEmpty)
                                  SizedBox(
                                    height: 120,
                                    child: ListView.builder(
                                      scrollDirection: Axis.horizontal,
                                      itemCount: frames.length,
                                      itemBuilder: (context, fidx) {
                                        return Padding(
                                          padding: const EdgeInsets.all(8.0),
                                          child: ClipRRect(
                                            borderRadius: BorderRadius.circular(
                                              12,
                                            ),
                                            child: Image.file(frames[fidx]),
                                          ),
                                        );
                                      },
                                    ),
                                  ),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

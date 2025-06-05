import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import 'maskinator.dart';

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
      _log += "$msg\n";
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

      setState(() {
        _extractionHistory.clear();
      });

      setState(() {
        _videoFile = videoFile;
        _log = '';
      });
      _appendLog('Selected video: ${videoFile.path}');

      processVideo();
    }
  }

  Future<void> processVideo() async {
    final List<File> frames = await _extractFrames();
    //_segmentFrames(frames);
    Maskinator fastSam = Maskinator();
    await fastSam.segment_images(frames);

    setState(() {
      if (_extractionHistory.isNotEmpty) {
        final last = _extractionHistory.first;
        last['log'] =
            '${last['log'] as String}Finished segmenting all images';
      }
    });
  }

  Future<List<File>> _extractFrames() async {
    if (_videoFile == null) return [];
    setState(() => _extracting = true);
    var stopwatch = Stopwatch()..start();

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

    stopwatch.stop();
    final elapsedMs = stopwatch.elapsedMilliseconds;
    print('Extracting frames took $elapsedMs ms\n');
    setState(() {
      if (_extractionHistory.isNotEmpty) {
        final last = _extractionHistory.first;
        last['log'] = '${last['log'] as String}Extracting frames took $elapsedMs ms\n';
      }
    });

    return files;
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
    final outputTensors = interpreter.getOutputTensors();

    final output = {
      0: List<List<List<double>>>.filled(
        1,
        List<List<double>>.filled(
          outputTensors[0].shape[1],
          List<double>.filled(outputTensors[0].shape[2], 0),
        ),
      ),
      1: List<List<List<List<double>>>>.filled(
        1,
        List<List<List<double>>>.filled(
          outputTensors[1].shape[1],
          List<List<double>>.filled(
            outputTensors[1].shape[2],
            List<double>.filled(outputTensors[1].shape[3], 0),
          ),
        ),
      ),
    };
    print('BAJA Ouput[0] matrix shape: ${output[0]!.shape.toString()}');
    print('BAJA Output[1] matrix shape: ${output[1]!.shape.toString()}');

    var stopwatch = Stopwatch()..start();

    var outputs = [];
    for (var frame in frames) {
      final input = await _filePreprocess(frame);
      interpreter.runForMultipleInputs([
        [input],
      ], output);
      outputs.add(output);
    }

    stopwatch.stop();
    final elapsedMs = stopwatch.elapsedMilliseconds;
    print('Segmentation inference time: ${elapsedMs} ms');
    print('avg segment time: ${(elapsedMs / frames.length).round()} ms');

    stopwatch = Stopwatch()..start();
    //List<List<Map<String, dynamic>>> frameMasks = [];
    for (var o in outputs) {
      //List<Map<String, dynamic>> masks = await _model2masks(o);
      //frameMasks.add(masks);
      computeModel2masks(o);
    }

    stopwatch.stop();
    print('all _model2masks time: ${stopwatch.elapsedMilliseconds} ms');
    print(
      'avg _model2masks time: ${(stopwatch.elapsedMilliseconds / frames.length).round()} ms',
    );

    //_logMasks(frameMasks, frames);

    // Add inference time to the extraction log for the current video
    setState(() {
      if (_extractionHistory.isNotEmpty) {
        final last = _extractionHistory.first;
        last['log'] =
            '${last['log'] as String}Segmentation inference time: $elapsedMs ms\nAverage Segmentation time: ${(elapsedMs / frames.length).round()} ms\nmodel2mask time: ${stopwatch.elapsedMilliseconds} ms\nAvg model2mask time: ${(stopwatch.elapsedMilliseconds / frames.length).round()} ms';
      }
    });

    print('After setState');
  }

  Future<List<Map<String, dynamic>>> computeModel2masks(var output) async {
    return await compute(_model2masks, output);
  }

  /*
I/com.example.app(29728): Thread[2,tid=29748,WaitingInMainSignalCatcherLoop,Thread*=0xb400007353714f50,peer=0x14003c40,"Signal Catcher"]: reacting to signal 3
I/com.example.app(29728):
I/com.example.app(29728): Wrote stack traces to tombstoned
*/
  static List<Map<String, dynamic>> _model2masks(var output, [double mnconf = 0.25]) {
    // 0,3,1,2 - output[1]
    var protos = output[1]![0]; // 160,160,32
    int h = protos.length, w = protos[0].length, c = protos[0][0].length;
    // flatten protos
    List<Float32List> flatProtos = [];
    for (int i = 0; i < c; i++) {
      Float32List flat = Float32List(h * w);
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          flat[j * w + k] = protos[j][k][i];
        }
      }
      flatProtos.add(flat);
    }

    // transpose out0(37,8400) to (8400,37) and filter out where objectness/conf > mnconf, <- maybe add nms LATER DO IT
    // then seperate into masks(5:), xywh(:4)
    List<List<double>> out0 =
        output[0]![0]; // 37,8400 - x, y, width, height, objectness and 32 mask coefficiants
    List<Float32List> out0Transpose = transpose2dDouble(out0),
        xywh = [],
        maskCoeff = [];
    for (var i = 0; i < out0Transpose.shape[0]; i++) {
      if (out0Transpose[i][4] > mnconf) {
        xywh.add(out0Transpose[i].sublist(0, 4));
        maskCoeff.add(out0Transpose[i].sublist(5));
      }
    }
    List<Float32List> xyxy = wh2xy(xywh);

    // matrix multiply flatten protos with maskCoeff
    List<Float32List> flat_masks = dotProd2d(
      maskCoeff,
      flatProtos,
    ); // n, 160*160

    // unflatten masks
    // resize to 640x640
    // normalize mask to boolean > 0, find area, crop to xyxy
    List<Map<String, dynamic>> ret = [];
    for (int i = 0; i < xyxy.length; i++) {
      int x1 = xyxy[i][0].round();
      int y1 = xyxy[i][1].round();
      int x2 = xyxy[i][2].round();
      int y2 = xyxy[i][3].round();
      x1 = x1.clamp(0, 639);
      y1 = y1.clamp(0, 639);
      x2 = x2.clamp(0, 639);
      y2 = y2.clamp(0, 639);

      List<List<double>> mask = unflatten(flat_masks[i], h, w);
      List<List<double>> resizedMask = resizeMatrix(mask, 640, 640);
      List<List<bool>> boolMask = List.generate(
        640,
        (j) => List.generate(640, (k) {
          return resizedMask[j][k] > 0 &&
              j >= y1 &&
              j < y2 &&
              k >= x1 &&
              k < x2;
        }),
      );

      int area = 0;
      for (int j = y1; j < y2; j++) {
        for (int k = x1; k < x2; k++) {
          area += boolMask[j][k] ? 1 : 0;
        }
      }

      ret.add({
        'xyxy': [x1, y1, x2, y2],
        'mask': boolMask,
        'area': area,
      });
    }

    ret.sort((a, b) => (b['area']).compareTo(a['area']));
    return ret;
  }

  static List<Float32List> transpose2dDouble(List<List<double>> m) {
    if (m.isEmpty) return [];
    // assume all rows have the same number of columns
    int rows = m.length;
    int cols = m[0].length;

    List<Float32List> result = [];
    for (int j = 0; j < cols; j++) {
      Float32List newRow = Float32List(rows);
      for (int i = 0; i < rows; i++) {
        newRow[i] = m[i][j];
      }
      result.add(newRow);
    }
    return result;
  }

  static List<Float32List> dotProd2d(List<Float32List> a, List<Float32List> b) {
    if (a.isEmpty || b.isEmpty) return [];

    final int aRows = a.length;
    final int aCols = a[0].length;
    final int bRows = b.length;
    final int bCols = b[0].length;

    // Check if matrices can be multiplied
    if (aCols != bRows) {
      throw ArgumentError(
        'Matrix dimensions do not match for multiplication: a[$aRows,$aCols] x b[$bRows,$bCols]',
      );
    }

    // Initialize the result matrix
    List<Float32List> result = List.generate(aRows, (_) => Float32List(bCols));

    // Perform matrix multiplication
    // Optimize by calculating in column-major order for better cache hits
    for (int i = 0; i < aRows; i++) {
      final Float32List rowA = a[i];
      final Float32List rowResult = result[i];

      for (int k = 0; k < bRows; k++) {
        final Float32List rowB = b[k];
        final double aik = rowA[k];

        for (int j = 0; j < bCols; j++) {
          rowResult[j] += aik * rowB[j];
        }
      }
    }

    return result;
  }

  static List<Float32List> wh2xy(List<Float32List> xywh) {
    List<Float32List> xyxy = [];
    for (var a in xywh) {
      Float32List xy = Float32List(4);
      xy[0] = a[0] - a[2] / 2;
      xy[1] = a[1] - a[3] / 2;
      xy[2] = a[0] - a[2] / 2;
      xy[3] = a[1] - a[3] / 2;
      xyxy.add(xy);
    }
    return xyxy;
  }

  static List<List<double>> unflatten(Float32List flat, int h, int w) {
    List<List<double>> matrix = List.generate(
      h,
      (i) => List.generate(w, (j) => flat[i * w + j].toDouble()),
    );
    return matrix;
  }

  static List<List<double>> resizeMatrix(
    List<List<double>> matrix,
    int newHeight,
    int newWidth,
  ) {
    if (matrix.isEmpty) return [];
    int oldHeight = matrix.length;
    int oldWidth = matrix[0].length;

    List<List<double>> resized = List.generate(
      newHeight,
      (i) => List.generate(newWidth, (j) {
        // Nearest neighbor resizing
        int srcX = (j * oldWidth / newWidth).round();
        int srcY = (i * oldHeight / newHeight).round();
        srcX = srcX.clamp(0, oldWidth - 1);
        srcY = srcY.clamp(0, oldHeight - 1);
        return matrix[srcY][srcX];
      }),
    );
    return resized;
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
                            width: 330, // 50% larger than 220
                            height: 330, // 50% larger than 220
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
                                  // ignore: deprecated_member_use
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
                                    height: 180,
                                    child: ListView.builder(
                                      scrollDirection: Axis.horizontal,
                                      itemCount: frames.length,
                                      itemBuilder: (context, fidx) {
                                        return Padding(
                                          padding: const EdgeInsets.all(8.0),
                                          child: ClipRRect(
                                            borderRadius: BorderRadius.circular(
                                              6,
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

import 'dart:io';
import 'dart:typed_data';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// SamInference
class SamInference {
  /// Instance of Interpreter
  late Interpreter _interpreter;

  static const String MODEL_FILE_NAME = "assets/FastSam-X.tflite";

  var outputBuff;

  /// Loads interpreter from asset
  Future<void> loadModel({Interpreter? interpreter}) async {
    try {
      final gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
      final delegate = switch (Platform.operatingSystem) {
        "ios" => GpuDelegate(),
        "android" => gpuDelegateV2,
        _ => XNNPackDelegate(),
      };

      InterpreterOptions options = InterpreterOptions()
        ..addDelegate(delegate)
        ..threads = 4;

      _interpreter = interpreter ?? await Interpreter.fromAsset(MODEL_FILE_NAME, options: options);
      print("Interpreter loaded successfully: $_interpreter");

      var outputTensors = _interpreter.getOutputTensors();

      outputBuff = {
        0: List<List<List<double>>>.filled(
          1,
          List<List<double>>.filled(outputTensors[0].shape[1], List<double>.filled(outputTensors[0].shape[2], 0)),
        ),
        1: List<List<List<List<double>>>>.filled(
          1,
          List<List<List<double>>>.filled(
            outputTensors[1].shape[1],
            List<List<double>>.filled(outputTensors[1].shape[2], List<double>.filled(outputTensors[1].shape[3], 0)),
          ),
        ),
      };
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  List<List<Object>> _filePreprocess(File image) {
    final bytes = image.readAsBytesSync();
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
    return [
      [imageMatrix],
    ];
  }

  /// Runs object detection on the input image
  Map<String, dynamic> predict(File image) {
    if (_interpreter == null) {
      print("Interpreter not initialized");
      return {};
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    final inputTensor = _filePreprocess(image);

    var preProcessElapsedTime = DateTime.now().millisecondsSinceEpoch - preProcessStart;
    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    _interpreter.runForMultipleInputs(inputTensor, outputBuff);

    var inferenceTimeElapsed = DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;
    var postStart = DateTime.now().millisecondsSinceEpoch;

    var ann = _model2masks(outputBuff);

    var postTime = DateTime.now().millisecondsSinceEpoch - postStart;

    return {
      "ann": ann,
      "stats": {"pre": preProcessElapsedTime, "inf": inferenceTimeElapsed, "post": postTime},
    };
  }

  List<Map<String, dynamic>> _model2masks(var output, [double mnconf = 0.70]) {
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
    List<List<double>> out0 = output[0]![0]; // 37,8400 - x, y, width, height, objectness and 32 mask coefficiants
    List<Float32List> out0Transpose = _transpose2dDouble(out0), xywh = [], maskCoeff = [];
    // Store confidence values for each detection
    List<double> confidences = [];
    for (var i = 0; i < out0Transpose.length; i++) {
      if (out0Transpose[i][4] > mnconf) {
        xywh.add(out0Transpose[i].sublist(0, 4));
        maskCoeff.add(out0Transpose[i].sublist(5));
        confidences.add(out0Transpose[i][4]); // Store confidence score
      }
    }
    List<Float32List> xyxy = _wh2xy(xywh);

    // matrix multiply flatten protos with maskCoeff
    List<Float32List> flat_masks = _dotProd2d(maskCoeff, flatProtos); // n, 160*160

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

      List<List<double>> mask = _unflatten(flat_masks[i], h, w);
      List<List<double>> resizedMask = _resizeMatrix(mask, 640, 640);
      List<List<bool>> boolMask = List.generate(
        640,
        (j) => List.generate(640, (k) {
          return resizedMask[j][k] > 0 && j >= y1 && j < y2 && k >= x1 && k < x2;
        }),
      );

      int marea = 0;
      for (int j = y1; j < y2; j++) {
        for (int k = x1; k < x2; k++) {
          marea += boolMask[j][k] ? 1 : 0;
        }
      }

      int area = (x2 - x1) * (y2 - y1);

      ret.add({
        'xyxy': [x1, y1, x2, y2],
        'mask': boolMask,
        'marea': marea,
        'area': area,
        'confidence': confidences[i],
      });
    }

    ret.sort((a, b) => b['confidence'].compareTo(a['confidence']));
    var nms = applyNMS(ret, 0.90);

    nms.sort((a, b) => b['area'].compareTo(a['area']));
    return nms;
  }

  List<Float32List> _transpose2dDouble(List<List<double>> m) {
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

  List<Float32List> _dotProd2d(List<Float32List> a, List<Float32List> b) {
    if (a.isEmpty || b.isEmpty) return [];

    final int aRows = a.length;
    final int aCols = a[0].length;
    final int bRows = b.length;
    final int bCols = b[0].length;

    // Check if matrices can be multiplied
    if (aCols != bRows) {
      throw ArgumentError('Matrix dimensions do not match for multiplication: a[$aRows,$aCols] x b[$bRows,$bCols]');
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

  List<Float32List> _wh2xy(List<Float32List> xywh) {
    List<Float32List> xyxy = [];
    for (var a in xywh) {
      Float32List xy = Float32List(4);
      xy[0] = a[0] - a[2] / 2;
      xy[1] = a[1] - a[3] / 2;
      xy[2] = a[0] + a[2] / 2;
      xy[3] = a[1] + a[3] / 2;
      xyxy.add(xy);
    }
    return xyxy;
  }

  List<List<double>> _unflatten(Float32List flat, int h, int w) {
    List<List<double>> matrix = List.generate(h, (i) => List.generate(w, (j) => flat[i * w + j].toDouble()));
    return matrix;
  }

  List<List<double>> _resizeMatrix(List<List<double>> matrix, int newHeight, int newWidth) {
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

  Interpreter get interpreter => _interpreter;

  // Applies Non-Maximum Suppression to the masks
  List<Map<String, dynamic>> applyNMS(List<Map<String, dynamic>> masks, double iouThreshold) {
    // If no masks or only one, return as is
    if (masks.length <= 1) return masks;

    // Sort by confidence (already done in _model2masks but ensuring here)
    masks.sort((a, b) => b['confidence'].compareTo(a['confidence']));

    List<Map<String, dynamic>> result = [];
    List<bool> suppressed = List.filled(masks.length, false);

    for (int i = 0; i < masks.length; i++) {
      if (suppressed[i]) continue;

      result.add(masks[i]);

      for (int j = i + 1; j < masks.length; j++) {
        if (suppressed[j]) continue;

        double iou = calculateIoU(masks[i], masks[j]);
        if (iou >= iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return result;
  }

  // Calculates IoU (Intersection over Union) between two masks
  double calculateIoU(Map<String, dynamic> maskA, Map<String, dynamic> maskB) {
    // Extract bounding boxes
    List<int> boxA = maskA['xyxy'];
    List<int> boxB = maskB['xyxy'];

    // Calculate intersection
    int xA = max(boxA[0], boxB[0]);
    int yA = max(boxA[1], boxB[1]);
    int xB = min(boxA[2], boxB[2]);
    int yB = min(boxA[3], boxB[3]);

    // If boxes don't overlap
    if (xB <= xA || yB <= yA) return 0.0;

    // Instead of just using bounding boxes, use the actual mask for accurate IoU
    int intersectionArea = 0;
    List<List<bool>> maskAData = maskA['mask'];
    List<List<bool>> maskBData = maskB['mask'];

    for (int y = yA; y < yB; y++) {
      for (int x = xA; x < xB; x++) {
        if (maskAData[y][x] && maskBData[y][x]) {
          intersectionArea++;
        }
      }
    }

    // Calculate union area: sum of both mask areas minus intersection
    int unionArea = maskA['marea'] + maskB['marea'] - intersectionArea;

    // Return IoU
    return unionArea > 0 ? intersectionArea / unionArea : 0.0;
  }
}

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Maskinator {
  Future<List<List<Map<String, dynamic>>>> segment_images(List<File> images) async {
    //List<Map<int, dynamic>> model_outputs = await compute(_sam_inference, images);
    List<Map<int, dynamic>> model_outputs = await _sam_inference(images);

    var stopwatch = Stopwatch()..start();

    List<List<Map<String, dynamic>>> masks = await computeAllOutputs2Masks(model_outputs);

    stopwatch.stop();
    final elapsedMs = stopwatch.elapsedMilliseconds;
    print('_model2masks time: ${elapsedMs} ms');
    print('avg _model2masks time: ${(elapsedMs / images.length).round()} ms');

    return masks;
  }

  static Future<List<Map<int, dynamic>>> _sam_inference(List<File> images, [String model_asset_path = 'assets/FastSam-X.tflite']) async {
    final gpuDelegateV2 = GpuDelegateV2(options: GpuDelegateOptionsV2());
    final delegate = switch (Platform.operatingSystem) {
      "ios" => GpuDelegate(),
      "android" => gpuDelegateV2,
      _ => XNNPackDelegate(),
    };

    final interpreter = await Interpreter.fromAsset(
      model_asset_path,
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

    List<Map<int, dynamic>> outputs = [];
    for (var frame in images) {
      final input = await _filePreprocess(frame);
      interpreter.runForMultipleInputs([
        [input],
      ], output);
      outputs.add(output);
    }

    stopwatch.stop();
    final elapsedMs = stopwatch.elapsedMilliseconds;
    print('Segmentation inference time: ${elapsedMs} ms');
    print('avg segment time: ${(elapsedMs / images.length).round()} ms');

    return outputs;
  }

  static Future<List<List<Object>>?> _filePreprocess(File image) async {
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

  Future<List<Map<String, dynamic>>> computeModel2masks(var output) async {
    return await compute(_model2masks, output);
  }

  Future<List<List<Map<String, dynamic>>>> computeAllOutputs2Masks(var outputs) async {
    return await compute(_allOutputs2Masks, outputs);
  }

  static List<List<Map<String, dynamic>>> _allOutputs2Masks(var model_outputs) {
    List<List<Map<String, dynamic>>> imageMasks = List.filled(model_outputs.length, []);
    for (var i = 0; i < model_outputs.length; i++) {
      imageMasks[i] = _model2masks(model_outputs[i]);
    }
    return imageMasks;
  }

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
    List<Float32List> out0Transpose = _transpose2dDouble(out0),
        xywh = [],
        maskCoeff = [];
    for (var i = 0; i < out0Transpose.shape[0]; i++) {
      if (out0Transpose[i][4] > mnconf) {
        xywh.add(out0Transpose[i].sublist(0, 4));
        maskCoeff.add(out0Transpose[i].sublist(5));
      }
    }
    List<Float32List> xyxy = _wh2xy(xywh);

    // matrix multiply flatten protos with maskCoeff
    List<Float32List> flat_masks = _dotProd2d(
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

      List<List<double>> mask = _unflatten(flat_masks[i], h, w);
      List<List<double>> resizedMask = _resizeMatrix(mask, 640, 640);
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

  static List<Float32List> _transpose2dDouble(List<List<double>> m) {
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

  static List<Float32List> _dotProd2d(List<Float32List> a, List<Float32List> b) {
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

  static List<Float32List> _wh2xy(List<Float32List> xywh) {
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

  static List<List<double>> _unflatten(Float32List flat, int h, int w) {
    List<List<double>> matrix = List.generate(
      h,
      (i) => List.generate(w, (j) => flat[i * w + j].toDouble()),
    );
    return matrix;
  }

  static List<List<double>> _resizeMatrix(
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
}

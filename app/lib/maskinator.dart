import 'dart:io';

import 'package:flutter/foundation.dart';

import 'sam_inference.dart';

class Maskinator {
  Maskinator() {
    samInf = SamInference();
  }

  late SamInference samInf;

  Future<void> init() async {
    await samInf.loadModel();
  }

  Future<List<Map<String, dynamic>>> runBatchIso(List<File> images) async {
    await init();
    return await compute(runBatchInference, {'images': images, 'samInf': samInf});
  }

  Future<List<Map<String, dynamic>>> runBatchInference(Map<String, dynamic> data) async {
    await data['samInf'].loadModel();
    final List<Map<String, dynamic>> results = [];
    for (var image in data['images']) {
      final result = await runModelIsolate(image);
      results.add(result);
    }
    return results;
  }

  Future<Map<String, dynamic>> runModelIsolate(File image) async {
    return await compute(inference, {'image': image, 'interpreterAddy': samInf.interpreter.address, 'samInf': samInf});
  }

  static Future<Map<String, dynamic>> inference(Map<String, dynamic> infData) async {
    return await infData['samInf'].predict(infData['image']);
  }
}

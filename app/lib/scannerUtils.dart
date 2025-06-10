import 'dart:typed_data';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';

class ScannerUtils {
  List<Map<String, dynamic>> filterByArea(List<Map<String, dynamic>> masks, double minPer, double maxPer, double fpct) {
    List<List<bool>> boolMask = masks[0]['ann'][0]['mask'];
    int frameSize = boolMask.length * boolMask[0].length;
    List<Map<String, dynamic>> filteredMasks = [];
    for (var frame in masks) {
      List<Map<String, dynamic>> filteredAnn = frame['ann'].where((ann) {
        int area = ann['area'];
        int numFalses = area - (ann['marea'] as int);
        return area >= minPer * frameSize && area <= maxPer * frameSize && numFalses < fpct * area;
      }).toList();
      filteredMasks.add({'ann': filteredAnn, 'stats': frame['stats']});
    }
    return filteredMasks;
  }
}

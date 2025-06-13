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

  bool mono(List<List<bool>> bmask, List<int> xyxy, int rnge, double slope) {
    List<List<int>> firstTrue = [];
    List<int> fw = List.filled(bmask.length, bmask[0].length - 1), bc = List.filled(bmask.length, 0);
    for (var i = 0; i < bmask.length; i++) {
      var lt = -1;
      for (var j = 0; j < bmask[0].length; j++) {
        if (bmask[i][j] && lt == -1) fw[i] = j;
        if (bmask[i][j]) lt = j;
      }
      if (lt != -1) bc[i] = lt;
    }
    for (var i = 0; i < bc.length; i++) {
      bc[i] = bmask[0].length - 1 - bc[i];
    }
    firstTrue.add(fw);
    firstTrue.add(bc);

    fw = List.filled(bmask[0].length, bmask.length - 1);
    bc = List.filled(bmask[0].length, 0);

    for (var i = 0; i < bmask[0].length; i++) {
      var lt = -1;
      for (var j = 0; j < bmask.length; j++) {
        if (bmask[j][i] && lt == -1) fw[i] = j;
        if (bmask[j][i]) lt = j;
      }
      if (lt != -1) bc[i] = lt;
    }
    for (var i = 0; i < bc.length; i++) {
      bc[i] = bmask.length - 1 - bc[i];
    }
    firstTrue.add(fw);
    firstTrue.add(bc);

    for (var ft in firstTrue) {
      bool fi = false;
      for (var j = 0; j < ft.length - rnge; j++) {
        if (ft[j + rnge] > ft[j] + (slope * rnge)) {
          fi = true;
        } else if (ft[j + rnge] <= ft[j] - (slope * rnge) && fi) {
          return false;
        }
      }
    }

    return true;
  }

  List<Map<String, dynamic>> filterByMono(List<Map<String, dynamic>> masks, int rnge, double slope) {
    List<Map<String, dynamic>> filteredMasks = [];
    for (var frame in masks) {
      List<Map<String, dynamic>> filteredAnn = frame['ann'].where((ann) {
        return mono(ann['mask'], ann['xyxy'], rnge, slope);
      }).toList();
      filteredMasks.add({'ann': filteredAnn, 'stats': frame['stats']});
    }
    return filteredMasks;
  }

  List<Map<String, dynamic>> filterCoveredMasks(List<Map<String, dynamic>> masks, int threshold) {
    // filter masks such that if any two masks have the any two corners of their bounding boxes within threshold pixels of each other, and are the same two corners(ie xyxy[0] and xyxy[0] and xyxy[2] and xyxy[2]), then remove the mask with the larger area
    List<bool> goodMasks = List.filled(masks.length, true);
    for (var i = 0; i < masks.length; i++) {
      for (var j = i + 1; j < masks.length; j++) {
        List<int> xyxy1 = masks[i]['xyxy'];
        List<int> xyxy2 = masks[j]['xyxy'];

        num closeCorners =
            (((xyxy1[0] - xyxy2[0]).abs() <= threshold) ? 1 : 0) +
            (((xyxy1[2] - xyxy2[2]).abs() <= threshold) ? 1 : 0) +
            (((xyxy1[1] - xyxy2[1]).abs() <= threshold) ? 1 : 0) +
            (((xyxy1[3] - xyxy2[3]).abs() <= threshold) ? 1 : 0);

        if (closeCorners >= 3) {
          int area1 = masks[i]['area'];
          int area2 = masks[j]['area'];
          if (area1 > area2) {
            goodMasks[i] = false;
          } else {
            goodMasks[j] = false;
          }
        }
      }
    }

    List<Map<String, dynamic>> filteredMasks = [];
    for (var i = 0; i < masks.length; i++) {
      if (goodMasks[i]) {
        filteredMasks.add(masks[i]);
      }
    }
    return filteredMasks;
  }

  List<Map<String, dynamic>> filterCoveredAll(List<Map<String, dynamic>> frames, bool isNotebook, int threshold) {
    for (var i = 0; i < frames.length; i++) {
      frames[i]['ann'] = filterCoveredMasks(frames[i]['ann'], threshold);
    }
    return frames;
  }

  List<Map<String, dynamic>> finalNumber(List<Map<String, dynamic>> frames, bool isNotebook) {
    List<Map<String, dynamic>> ret = [];
    for (var frame in frames) {
      if (frame['ann'].length == (isNotebook ? 2 : 1)) {
        ret.add(frame);
      } else {
        ret.add({'ann': [], 'stats': frame['stats']});
      }
    }
    return ret;
  }

  List<Map<String, dynamic>> filter(
    List<Map<String, dynamic>> modelOut,
    bool isNotebook, [
    double minPer = 0.10,
    double maxPer = 0.70,
    double fpct = 0.10,
    int rnge = 10,
    double slope = 0.50,
    int coveredThreshold = 40,
  ]) {
    return finalNumber(
      filterByMono(filterCoveredAll(filterByArea(modelOut, minPer, maxPer, fpct), isNotebook, coveredThreshold), rnge, slope),
      isNotebook,
    );
  }

  // find final frames with 1(if !isNotebook) or 2 (if isNotebook) masks, if many are valid in a row, choose the frame with the largest 'area'
  List<Map<String, dynamic>> finalFrames(List<Map<String, dynamic>> frames) {
    print('Final Frames');
    List<Map<String, dynamic>> ff = List.generate(frames.length, (i) => {'ann': [], 'stats': frames[i]['stats']}),
      cur = [];
    for (var i = 0; i < frames.length; i++) {
      if (frames[i]['ann'].isNotEmpty) {
        int tarea = 0;
        for (var f in frames[i]['ann']) tarea += f['area'] as int;
        cur.add({'ann': List.from(frames[i]['ann']), 'tarea': tarea, 'i': i});
      }

      if ((frames[i]['ann'].isEmpty || i == frames.length-1) && cur.isNotEmpty) {
        cur.sort((a, b) => b['tarea'].compareTo(a['tarea']));
        ff[cur.first['i']]['ann'] = List.from(cur.first['ann']);
        cur.clear();
      }
    }
    return ff;
  }
}

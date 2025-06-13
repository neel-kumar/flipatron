import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;

import 'maskinator.dart';

class Scanner {
  final Maskinator _maskinator = Maskinator();
  bool _isInitialized = false;
  
  // Status reporting
  Function(String)? _progressCallback;
  Function(List<Map<String, dynamic>>)? _frameCallback;

  // Constructor
  Scanner({Function(String)? onProgress, Function(List<Map<String, dynamic>>)? onFrameProcessed}) {
    _progressCallback = onProgress;
    _frameCallback = onFrameProcessed;
  }

  /// Initialize the model
  Future<void> init() async {
    if (!_isInitialized) {
      await _maskinator.init();
      _isInitialized = true;
    }
  }

  /// Log progress
  void _reportProgress(String message) {
    if (_progressCallback != null) {
      _progressCallback!(message);
    } else {
      print(message);
    }
  }

  /// Extract frames from a video file at given FPS
  Future<List<File>> videoToFrames(File videoFile, double fps) async {
    _reportProgress("Extracting frames at $fps FPS...");
    
    // Create a temp directory to store frames
    final dir = await getTemporaryDirectory();
    final framesDir = Directory('${dir.path}/frames');
    if (framesDir.existsSync()) {
      try {
        await framesDir.delete(recursive: true);
      } catch (e) {
        print("Error deleting frames directory: $e");
      }
    }
    await framesDir.create();
    
    // Get video info
    final videoFpsCmd = '-v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 "${videoFile.path}"';
    final videoFpsResult = await FFmpegKit.execute(videoFpsCmd);
    final videoFpsOutput = await videoFpsResult.getOutput();
    
    double videoFps = 30.0; // Default fallback
    if (videoFpsOutput != null && videoFpsOutput.isNotEmpty) {
      try {
        final parts = videoFpsOutput.trim().split('/');
        if (parts.length == 2) {
          final numerator = int.parse(parts[0]);
          final denominator = int.parse(parts[1]);
          videoFps = numerator / denominator;
        }
      } catch (e) {
        print("Error parsing video FPS: $e");
      }
    }
    
    _reportProgress("Video FPS: $videoFps");
    
    // Calculate frame interval
    final frameInterval = (videoFps / fps).round();
    
    // Extract frames
    final videoBase = videoFile.path.split(Platform.pathSeparator).last.split('.').first;
    final outPattern = '${framesDir.path}/${videoBase}_%05d.jpg';
    final cmd = '-i "${videoFile.path}" -vf fps=1/$frameInterval "${outPattern}"';
    
    var stopwatch = Stopwatch()..start();
    await FFmpegKit.execute(cmd);
    stopwatch.stop();
    
    // Get the extracted frames
    final RegExp framePattern = RegExp('^${RegExp.escape(videoBase)}_\\d{5}\\.jpg');
    final files = framesDir
        .listSync()
        .whereType<File>()
        .where((f) => framePattern.hasMatch(f.path.split(Platform.pathSeparator).last))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    
    _reportProgress("Extracted ${files.length} frames in ${stopwatch.elapsedMilliseconds}ms");
    return files;
  }

  /// Find bounding box from a boolean mask
  List<int>? findBoundingBox(List<List<bool>> mask) {
    if (mask.isEmpty) return null;
    
    int minRow = mask.length;
    int maxRow = -1;
    int minCol = mask[0].length;
    int maxCol = -1;
    
    bool hasTrue = false;
    
    for (int i = 0; i < mask.length; i++) {
      for (int j = 0; j < mask[i].length; j++) {
        if (mask[i][j]) {
          hasTrue = true;
          minRow = math.min(minRow, i);
          maxRow = math.max(maxRow, i);
          minCol = math.min(minCol, j);
          maxCol = math.max(maxCol, j);
        }
      }
    }
    
    if (!hasTrue) return null; // Handle empty mask case
    
    // Return [x, y, width, height]
    return [minCol, minRow, maxCol - minCol, maxRow - minRow];
  }

  /// Filter out covered pages
  List<Map<String, dynamic>> filterCoveredPages(
      List<Map<String, dynamic>> pages, double thresholdPercent) {
    if (pages.isEmpty) return [];
    if (pages.length <= 2) return pages;

    final List<Map<String, dynamic>> filtered = [];
    
    // Create a mask for the full image, initially all false
    List<List<bool>> imgMask = List.generate(
      (pages[0]['mask'] as List<List<bool>>).length,
      (_) => List.generate(
        (pages[0]['mask'][0] as List<bool>).length,
        (_) => false
      )
    );
    
    // Sort pages by area (ascending so we process smaller areas first)
    final sortedPages = List<Map<String, dynamic>>.from(pages)
      ..sort((a, b) => a['area'].compareTo(b['area']));
    
    for (var page in sortedPages) {
      final List<int> bbox = page['xyxy'] as List<int>;
      final List<List<bool>> pageMask = page['mask'] as List<List<bool>>;
      
      int overlappingPixels = 0;
      int totalPixels = 0;
      
      // Count how many pixels in this page are already covered by previous pages
      for (int y = bbox[1]; y < bbox[3]; y++) {
        for (int x = bbox[0]; x < bbox[2]; x++) {
          if (y < pageMask.length && x < pageMask[y].length && pageMask[y][x]) {
            totalPixels++;
            if (y < imgMask.length && x < imgMask[y].length && imgMask[y][x]) {
              overlappingPixels++;
            }
          }
        }
      }
      
      // If less than threshold% of the page is covered, include it
      if (totalPixels > 0 && overlappingPixels < totalPixels * thresholdPercent) {
        filtered.add(page);
        
        // Update the mask with this page's pixels
        for (int y = bbox[1]; y < bbox[3]; y++) {
          for (int x = bbox[0]; x < bbox[2]; x++) {
            if (y < pageMask.length && x < pageMask[y].length && pageMask[y][x] &&
                y < imgMask.length && x < imgMask[y].length) {
              imgMask[y][x] = true;
            }
          }
        }
      }
    }
    
    return filtered;
  }

  /// Find pages in the masks
  List<Map<String, dynamic>> findPages(
      List<Map<String, dynamic>> masks, 
      double areaPercent, 
      double areaMax,
      double fillPercent) {
    if (masks.isEmpty) return [];
    
    List<Map<String, dynamic>> pages = [];
    final int frameSize = (masks[0]['mask'] as List<List<bool>>).length * 
                        (masks[0]['mask'] as List<List<bool>>)[0].length;
    
    for (var mask in masks) {
      final int area = mask['area'] as int;
      
      // Check if the area is within bounds
      if (area > frameSize * areaPercent && area < frameSize * areaMax) {
        List<int> bbox = mask['xyxy'] as List<int>;
        List<List<bool>> pageMask = mask['mask'] as List<List<bool>>;
        
        // Count false pixels inside the bbox to check how filled the rectangle is
        int falsePixels = 0;
        int totalPixels = 0;
        
        for (int y = bbox[1]; y < bbox[3]; y++) {
          for (int x = bbox[0]; x < bbox[2]; x++) {
            if (y < pageMask.length && x < pageMask[0].length) {
              totalPixels++;
              if (!pageMask[y][x]) {
                falsePixels++;
              }
            }
          }
        }
        
        // If the box is filled enough (not too many holes)
        if (totalPixels > 0 && falsePixels < totalPixels * fillPercent) {
          pages.add(mask);
        }
      }
    }
    
    return pages;
  }

  /// Check if the mask has a monotonic edge (to filter out noise/obstructions)
  bool isMonotonic(Map<String, dynamic> mask, int range, double slopeThreshold) {
    List<List<bool>> segmentation = mask['mask'] as List<List<bool>>;
    
    // Check in 4 directions (left-to-right, right-to-left, top-to-bottom, bottom-to-top)
    for (int rotation = 0; rotation < 4; rotation++) {
      List<int> firstTruePosition = [];
      
      // Find the first true pixel in each row
      for (int i = 0; i < segmentation.length; i++) {
        int firstTrue = -1;
        for (int j = 0; j < segmentation[i].length; j++) {
          if (segmentation[i][j]) {
            firstTrue = j;
            break;
          }
        }
        firstTruePosition.add(firstTrue != -1 ? firstTrue : segmentation[i].length);
      }
      
      // Check for monotonicity with slope threshold
      bool increasing = false;
      for (int i = 0; i < firstTruePosition.length - range; i++) {
        if ((firstTruePosition[i + range] - firstTruePosition[i]).abs() / range > slopeThreshold) {
          if (firstTruePosition[i + range] > firstTruePosition[i]) {
            increasing = true;
          } else if (increasing) {
            return false;  // If it was increasing and now decreasing
          }
        }
      }
      
      // Rotate the segmentation 90 degrees clockwise for the next check
      segmentation = _rotate90Clockwise(segmentation);
    }
    
    return true;
  }

  /// Helper function to rotate a boolean matrix 90 degrees clockwise
  List<List<bool>> _rotate90Clockwise(List<List<bool>> matrix) {
    if (matrix.isEmpty) return [];
    
    int rows = matrix.length;
    int cols = matrix[0].length;
    
    List<List<bool>> rotated = List.generate(
      cols, 
      (_) => List.generate(rows, (_) => false)
    );
    
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        rotated[j][rows - 1 - i] = matrix[i][j];
      }
    }
    
    return rotated;
  }

  /// Find ranges of visible pages and select the middle frame from each range
  List<int> findFinalFrames(List<bool> goodPageFrames) {
    List<int> finalFrames = [];
    int i = 0;
    
    while (i < goodPageFrames.length) {
      // Skip frames without good pages
      while (i < goodPageFrames.length && !goodPageFrames[i]) {
        i++;
      }
      
      if (i >= goodPageFrames.length) break;
      
      int start = i;
      
      // Find the end of this continuous sequence of good pages
      while (i + 1 < goodPageFrames.length && goodPageFrames[i + 1]) {
        i++;
      }
      
      // Take the middle frame from this range
      finalFrames.add((start + i) ~/ 2);
      
      i++;
    }
    
    return finalFrames;
  }

  /// Process the video file and return the PDF as bytes
  Future<Map<String, dynamic>> processVideo(File videoFile, double fps, bool isNotebook, {
    double areaPercent = 0.20,
    double areaMax = 0.50,
    double fillPercent = 0.10,
    double coverThreshold = 0.20,
    int monoRange = 10,
    double monoSlope = 0.50
  }) async {
    await init();

    // Extract frames from video
    final frames = await videoToFrames(videoFile, fps);
    _reportProgress("Processing ${frames.length} frames...");
    
    // Process frames to detect pages
    List<List<Map<String, dynamic>>> pagesInFrames = [];
    List<Map<String, dynamic>> processingResults = [];
    
    // Process frames with FastSAM model
    var stopwatch = Stopwatch()..start();
    processingResults = await _maskinator.runBatchIso(frames);
    stopwatch.stop();
    _reportProgress("Segmentation completed in ${stopwatch.elapsedMilliseconds}ms");
    
    // Find pages in each frame
    List<bool> goodPageFrames = [];
    
    for (int i = 0; i < processingResults.length; i++) {
      final result = processingResults[i];
      final masks = result['ann'] as List<Map<String, dynamic>>;
      
      // Find possible pages in this frame
      List<Map<String, dynamic>> pages = findPages(masks, areaPercent, areaMax, fillPercent);
      
      // Filter out overlapping pages
      pages = filterCoveredPages(pages, coverThreshold);
      
      pagesInFrames.add(pages);
      
      // Check if this frame has good pages (right number and monotonic edges)
      bool isGoodFrame = false;
      if ((isNotebook && pages.length == 2) || (!isNotebook && pages.length == 1)) {
        isGoodFrame = pages.every((page) => isMonotonic(page, monoRange, monoSlope));
      }
      
      goodPageFrames.add(isGoodFrame);
      
      _reportProgress("Frame ${i+1}/${processingResults.length}: Found ${pages.length} pages");
      
      if (_frameCallback != null) {
        _frameCallback!([
          {'frameIndex': i, 'pages': pages, 'isGoodFrame': isGoodFrame}
        ]);
      }
    }

    // Find the best frames that represent unique pages
    List<int> finalFrameIndices = findFinalFrames(goodPageFrames);
    _reportProgress("Found ${finalFrameIndices.length} unique page views");
    
    // Create list of final pages
    List<img.Image> finalPages = [];
    List<Map<String, dynamic>> frameData = [];
    
    for (int idx in finalFrameIndices) {
      // Sort pages left to right if there are multiple
      pagesInFrames[idx].sort((a, b) => (a['xyxy'][0]).compareTo(b['xyxy'][0]));
      
      for (var page in pagesInFrames[idx]) {
        final List<int> bbox = page['xyxy'] as List<int>;
        
        // Load and crop the frame to the page boundaries
        final File frameFile = frames[idx];
        final bytes = frameFile.readAsBytesSync();
        final img.Image? decoded = img.decodeImage(bytes);
        
        if (decoded != null) {
          // Crop the page
          img.Image cropped = img.copyCrop(
            decoded,
            x: bbox[0],
            y: bbox[1],
            width: bbox[2] - bbox[0],
            height: bbox[3] - bbox[1]
          );
          
          // Add to final pages list
          finalPages.add(cropped);
          
          // Add frame data for display in the app
          frameData.add({
            'frameIndex': idx,
            'bbox': bbox,
            'pageImage': cropped
          });
        }
      }
    }
    
    // Create PDF document
    final pdfBytes = await _createPdf(finalPages);
    
    return {
      'pdfBytes': pdfBytes,
      'pages': finalPages,
      'frameData': frameData,
      'frameIndices': finalFrameIndices
    };
  }

  /// Create PDF document from the detected pages
  Future<Uint8List> _createPdf(List<img.Image> pages) async {
    final pdf = pw.Document();
    
    for (var page in pages) {
      final imgBytes = img.encodePng(page);
      final image = pw.MemoryImage(Uint8List.fromList(imgBytes));
      
      pdf.addPage(
        pw.Page(
          build: (pw.Context context) {
            return pw.Center(
              child: pw.Image(image),
            );
          },
        ),
      );
    }
    
    return pdf.save();
  }

  /// Save PDF to a file and return the path
  Future<String> savePdf(Uint8List pdfBytes, String fileName) async {
    try {
      // Get directory for saving the file
      Directory? dir;
      if (Platform.isAndroid) {
        dir = await getExternalStorageDirectory();  // for Android
      } else {
        dir = await getApplicationDocumentsDirectory(); // for iOS, desktop
      }
      
      if (dir == null) {
        throw Exception('Could not access storage directory');
      }
      
      // Ensure file extension is .pdf
      if (!fileName.toLowerCase().endsWith('.pdf')) {
        fileName = '$fileName.pdf';
      }
      
      final file = File('${dir.path}/$fileName');
      await file.writeAsBytes(pdfBytes);
      return file.path;
    } catch (e) {
      print('Error saving PDF: $e');
      throw Exception('Failed to save PDF: $e');
    }
  }
}
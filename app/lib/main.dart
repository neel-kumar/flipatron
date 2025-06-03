import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

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

      // Delete old saved frames, needs to be implemented

      setState(() {
        _videoFile = videoFile;
        _log = '';
      });
      _appendLog('Selected video: ${videoFile.path}');

      final List<File> frames = await _extractFrames();
      _segmentFrames(frames);
    }
  }

  Future<List<File>> _extractFrames() async {
    if (_videoFile == null) return [];
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

    return files;
    //_segmentFrames(files);
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
    print('BAJA Ouput tensors 2: ${outputTensors[2].shape.toString()}');
    print('BAJA Ouput tensors 3: ${outputTensors[3].shape.toString()}');
    print('BAJA Ouput tensors 4: ${outputTensors[4].shape.toString()}');
    print('BAJA Ouput tensors 5: ${outputTensors[5].shape.toString()}');
    print('BAJA Ouput tensors 6: ${outputTensors[6].shape.toString()}');
    print('BAJA Ouput tensors 7: ${outputTensors[7].shape.toString()}');
    print('BAJA Ouput tensors 8: ${outputTensors[8].shape.toString()}');
    print('BAJA Ouput tensors 9: ${outputTensors[9].shape.toString()}');
    print('BAJA Ouput tensors 10: ${outputTensors[10].shape.toString()}');
    print('BAJA Ouput tensors 11: ${outputTensors[11].shape.toString()}');
    print('BAJA Ouput tensors 12: ${outputTensors[12].shape.toString()}');
    print('BAJA Ouput tensors 13: ${outputTensors[13].shape.toString()}');
    print('BAJA Ouput tensors 14: ${outputTensors[14].shape.toString()}');
    print('BAJA Ouput tensors 15: ${outputTensors[15].shape.toString()}');
    print('BAJA Ouput tensors 16: ${outputTensors[16].shape.toString()}');
    print('BAJA Ouput tensors 17: ${outputTensors[17].shape.toString()}');
    print('BAJA Ouput tensors 18: ${outputTensors[18].shape.toString()}');
    print('BAJA Ouput tensors 19: ${outputTensors[19].shape.toString()}');
    print('BAJA Ouput tensors 20: ${outputTensors[20].shape.toString()}');
    print('BAJA Ouput tensors 21: ${outputTensors[21].shape.toString()}');
    print('BAJA Ouput tensors 22: ${outputTensors[22].shape.toString()}');
    print('BAJA Ouput tensors 23: ${outputTensors[23].shape.toString()}');
    print('BAJA Ouput tensors 24: ${outputTensors[24].shape.toString()}');
    print('BAJA Ouput tensors 25: ${outputTensors[25].shape.toString()}');
    print('BAJA Ouput tensors 26: ${outputTensors[26].shape.toString()}');
    print('BAJA Ouput tensors 27: ${outputTensors[27].shape.toString()}');
    print('BAJA Ouput tensors 28: ${outputTensors[28].shape.toString()}');
    print('BAJA Ouput tensors 29: ${outputTensors[29].shape.toString()}');
    print('BAJA Ouput tensors 30: ${outputTensors[30].shape.toString()}');
    print('BAJA Ouput tensors 31: ${outputTensors[31].shape.toString()}');
    print('BAJA Ouput tensors 32: ${outputTensors[32].shape.toString()}');
    print('BAJA Ouput tensors 33: ${outputTensors[33].shape.toString()}');
    print('BAJA Ouput tensors 34: ${outputTensors[34].shape.toString()}');
    print('BAJA Ouput tensors 35: ${outputTensors[35].shape.toString()}');
    print('BAJA Ouput tensors 36: ${outputTensors[36].shape.toString()}');
    print('BAJA Ouput tensors 37: ${outputTensors[37].shape.toString()}');
    print('BAJA Ouput tensors 38: ${outputTensors[38].shape.toString()}');
    print('BAJA Ouput tensors 39: ${outputTensors[39].shape.toString()}');
    print('BAJA Ouput tensors 40: ${outputTensors[40].shape.toString()}');
    print('BAJA Ouput tensors 41: ${outputTensors[41].shape.toString()}');
    print('BAJA Ouput tensors 42: ${outputTensors[42].shape.toString()}');
    print('BAJA Ouput tensors 43: ${outputTensors[43].shape.toString()}');
    print('BAJA Ouput tensors 44: ${outputTensors[44].shape.toString()}');
    print('BAJA Ouput tensors 45: ${outputTensors[45].shape.toString()}');
    print('BAJA Ouput tensors 46: ${outputTensors[46].shape.toString()}');
    print('BAJA Ouput tensors 47: ${outputTensors[47].shape.toString()}');
    print('BAJA Ouput tensors 48: ${outputTensors[48].shape.toString()}');
    print('BAJA Ouput tensors 49: ${outputTensors[49].shape.toString()}');
    print('BAJA Ouput tensors 50: ${outputTensors[50].shape.toString()}');
    print('BAJA Ouput tensors 51: ${outputTensors[51].shape.toString()}');
    print('BAJA Ouput tensors 52: ${outputTensors[52].shape.toString()}');
    print('BAJA Ouput tensors 53: ${outputTensors[53].shape.toString()}');
    print('BAJA Ouput tensors 54: ${outputTensors[54].shape.toString()}');
    print('BAJA Ouput tensors 55: ${outputTensors[55].shape.toString()}');
    print('BAJA Ouput tensors 56: ${outputTensors[56].shape.toString()}');
    print('BAJA Ouput tensors 57: ${outputTensors[57].shape.toString()}');
    print('BAJA Ouput tensors 58: ${outputTensors[58].shape toString()}');
    print('BAJA Ouput tensors 59: ${outputTensors[59].shape toString()}');
    print('BAJA Ouput tensors 60: ${outputTensors[60].shape toString()}');
    print('BAJA Ouput tensors 61: ${outputTensors[61].shape toString()}');
    print('BAJA Ouput tensors 62: ${outputTensors[62].shape toString()}');
    print('BAJA Ouput tensors 63: ${outputTensors[63].shape toString()}');
    print('BAJA Ouput tensors 64: ${outputTensors[64].shape toString()}');
    print('BAJA Ouput tensors 65: ${outputTensors[65].shape toString()}');
    print('BAJA Ouput tensors 66: ${outputTensors[66].shape toString()}');
    print('BAJA Ouput tensors 67: ${outputTensors[67].shape toString()}');
    print('BAJA Ouput tensors 68: ${outputTensors[68].shape toString()}');
    print('BAJA Ouput tensors 69: ${outputTensors[69].shape toString()}');
    print('BAJA Ouput tensors 70: ${outputTensors[70].shape toString()}');
    print('BAJA Ouput tensors 71: ${outputTensors[71].shape toString()}');
    print('BAJA Ouput tensors 72: ${outputTensors[72].shape toString()}');
    print('BAJA Ouput tensors 73: ${outputTensors[73].shape toString()}');
    print('BAJA Ouput tensors 74: ${outputTensors[74].shape toString()}');
    print('BAJA Ouput tensors 75: ${outputTensors[75].shape toString()}');
    print('BAJA Ouput tensors 76: ${outputTensors[76].shape toString()}');
    print('BAJA Ouput tensors 77: ${outputTensors[77].shape toString()}');
    print('BAJA Ouput tensors 78: ${outputTensors[78].shape toString()}');
    print('BAJA Ouput tensors 79: ${outputTensors[79].shape toString()}');
    print('BAJA Ouput tensors 80: ${outputTensors[80].shape toString()}');
    print('BAJA Ouput tensors 81: ${outputTensors[81].shape toString()}');
    print('BAJA Ouput tensors 82: ${outputTensors[82].shape toString()}');
    print('BAJA Ouput tensors 83: ${outputTensors[83].shape toString()}');
    print('BAJA Ouput tensors 84: ${outputTensors[84].shape toString()}');
    print('BAJA Ouput tensors 85: ${outputTensors[85].shape toString()}');
    print('BAJA Ouput tensors 86: ${outputTensors[86].shape toString()}');
    print('BAJA Ouput tensors 87: ${outputTensors[87].shape toString()}');
    print('BAJA Ouput tensors 88: ${outputTensors[88].shape toString()}');
    print('BAJA Ouput tensors 89: ${outputTensors[89].shape toString()}');
    print('BAJA Ouput tensors 90: ${outputTensors[90].shape toString()}');
    print('BAJA Ouput tensors 91: ${outputTensors[91].shape toString()}');
    print('BAJA Ouput tensors 92: ${outputTensors[92].shape toString()}');
    print('BAJA Ouput tensors 93: ${outputTensors[93].shape toString()}');
    print('BAJA Ouput tensors 94: ${outputTensors[94].shape toString()}');
    print('BAJA Ouput tensors 95: ${outputTensors[95].shape toString()}');
    print('BAJA Ouput tensors 96: ${outputTensors[96].shape toString()}');
    print('BAJA Ouput tensors 97: ${outputTensors[97].shape toString()}');
    print('BAJA Ouput tensors 98: ${outputTensors[98].shape toString()}');
    print('BAJA Ouput tensors 99: ${outputTensors[99].shape toString()}');
    print('BAJA Ouput tensors 100: ${outputTensors[100].shape toString()}');
    print('BAJA Ouput tensors 101: ${outputTensors[101].shape toString()}');
    print('BAJA Ouput tensors 102: ${outputTensors[102].shape toString()}');
    print('BAJA Ouput tensors 103: ${outputTensors[103].shape toString()}');
    print('BAJA Ouput tensors 104: ${outputTensors[104].shape toString()}');
    print('BAJA Ouput tensors 105: ${outputTensors[105].shape toString()}');
    print('BAJA Ouput tensors 106: ${outputTensors[106].shape toString()}');
    print('BAJA Ouput tensors 107: ${outputTensors[107].shape toString()}');
    print('BAJA Ouput tensors 108: ${outputTensors[108].shape toString()}');
    print('BAJA Ouput tensors 109: ${outputTensors[109].shape toString()}');
    print('BAJA Ouput tensors 110: ${outputTensors[110].shape toString()}');
    print('BAJA Ouput tensors 111: ${outputTensors[111].shape toString()}');
    print('BAJA Ouput tensors 112: ${outputTensors[112].shape toString()}');
    print('BAJA Ouput tensors 113: ${outputTensors[113].shape toString()}');
    print('BAJA Ouput tensors 114: ${outputTensors[114].shape toString()}');
    print('BAJA Ouput tensors 115: ${outputTensors[115].shape toString()}');
    print('BAJA Ouput tensors 116: ${outputTensors[116].shape toString()}');
    print('BAJA Ouput tensors 117: ${outputTensors[117].shape toString()}');
    print('BAJA Ouput tensors 118: ${outputTensors[118].shape toString()}');
    print('BAJA Ouput tensors 119: ${outputTensors[119].shape toString()}');
    print('BAJA Ouput tensors 120: ${outputTensors[120].shape toString()}');
    print('BAJA Ouput tensors 121: ${outputTensors[121].shape toString()}');
    print('BAJA Ouput tensors 122: ${outputTensors[122].shape toString()}');
    print('BAJA Ouput tensors 123: ${outputTensors[123].shape toString()}');
    print('BAJA Ouput tensors 124: ${outputTensors[124].shape toString()}');
    print('BAJA Ouput tensors 125: ${outputTensors[125].shape toString()}');
    print('BAJA Ouput tensors 126: ${outputTensors[126].shape toString()}');
    print('BAJA Ouput tensors 127: ${outputTensors[127].shape toString()}');
    print('BAJA Ouput tensors 128: ${outputTensors[128].shape toString()}');
    print('BAJA Ouput tensors 129: ${outputTensors[129].shape toString()}');
    print('BAJA Ouput tensors 130: ${outputTensors[130].shape toString()}');
    print('BAJA Ouput tensors 131: ${outputTensors[131].shape toString()}');
    print('BAJA Ouput tensors 132: ${outputTensors[132].shape toString()}');
    print('BAJA Ouput tensors 133: ${outputTensors[133].shape toString()}');
    print('BAJA Ouput tensors 134: ${outputTensors[134].shape toString()}');
    print('BAJA Ouput tensors 135: ${outputTensors[135].shape toString()}');
    print('BAJA Ouput tensors 136: ${outputTensors[136].shape toString()}');
    print('BAJA Ouput tensors 137: ${outputTensors[137].shape toString()}');
    print('BAJA Ouput tensors 138: ${outputTensors[138].shape toString()}');
    print('BAJA Ouput tensors 139: ${outputTensors[139].shape toString()}');
    print('BAJA Ouput tensors 140: ${outputTensors[140].shape toString()}');
    print('BAJA Ouput tensors 141: ${outputTensors[141].shape toString()}');
    print('BAJA Ouput tensors 142: ${outputTensors[142].shape toString()}');
    print('BAJA Ouput tensors 143: ${outputTensors[143].shape toString()}');
    print('BAJA Ouput tensors 144: ${outputTensors[144].shape toString()}');
    print('BAJA Ouput tensors 145: ${outputTensors[145].shape toString()}');
    print('BAJA Ouput tensors 146: ${outputTensors[146].shape toString()}');
    print('BAJA Ouput tensors 147: ${outputTensors[147].shape toString()}');
    print('BAJA Ouput tensors 148: ${outputTensors[148].shape toString()}');
    print('BAJA Ouput tensors 149: ${outputTensors[149].shape toString()}');
    print('BAJA Ouput tensors 150: ${outputTensors[150].shape toString()}');
    print('BAJA Ouput tensors 151: ${outputTensors[151].shape toString()}');
    print('BAJA Ouput tensors 152: ${outputTensors[152].shape toString()}');
    print('BAJA Ouput tensors 153: ${outputTensors[153].shape toString()}');
    print('BAJA Ouput tensors 154: ${outputTensors[154].shape toString()}');
    print('BAJA Ouput tensors 155: ${outputTensors[155].shape toString()}');
    print('BAJA Ouput tensors 156: ${outputTensors[156].shape toString()}');
    print('BAJA Ouput tensors 157: ${outputTensors[157].shape toString()}');
    print('BAJA Ouput tensors 158: ${outputTensors[158].shape toString()}');
    print('BAJA Ouput tensors 159: ${outputTensors[159].shape toString()}');
    print('BAJA Ouput tensors 160: ${outputTensors[160].shape toString()}');
    print('BAJA Ouput tensors 161: ${outputTensors[161].shape toString()}');
    print('BAJA Ouput tensors 162: ${outputTensors[162].shape toString()}');
    print('BAJA Ouput tensors 163: ${outputTensors[163].shape toString()}');
    print('BAJA Ouput tensors 164: ${outputTensors[164].shape toString()}');
    print('BAJA Ouput tensors 165: ${outputTensors[165].shape toString()}');
    print('BAJA Ouput tensors 166: ${outputTensors[166].shape toString()}');
    print('BAJA Ouput tensors 167: ${outputTensors[167].shape toString()}');
    print('BAJA Ouput tensors 168: ${outputTensors[168].shape toString()}');
    print('BAJA Ouput tensors 169: ${outputTensors[169].shape toString()}');
    print('BAJA Ouput tensors 170: ${outputTensors[170].shape toString()}');
    print('BAJA Ouput tensors 171: ${outputTensors[171].shape toString()}');
    print('BAJA Ouput tensors 172: ${outputTensors[172].shape toString()}');
    print('BAJA Ouput tensors 173: ${outputTensors[173].shape toString()}');
    print('BAJA Ouput tensors 174: ${outputTensors[174].shape toString()}');
    print('BAJA Ouput tensors 175: ${outputTensors[175].shape toString()}');
    print('BAJA Ouput tensors 176: ${outputTensors[176].shape toString()}');
    print('BAJA Ouput tensors 177: ${outputTensors[177].shape toString()}');
    print('BAJA Ouput tensors 178: ${outputTensors[178].shape toString()}');
    print('BAJA Ouput tensors 179: ${outputTensors[179].shape toString()}');
    print('BAJA Ouput tensors 180: ${outputTensors[180].shape toString()}');
    print('BAJA Ouput tensors 181: ${outputTensors[181].shape toString()}');
    print('BAJA Ouput tensors 182: ${outputTensors[182].shape toString()}');
    print('BAJA Ouput tensors 183: ${outputTensors[183].shape toString()}');
    print('BAJA Ouput tensors 184: ${outputTensors[184].shape toString()}');
    print('BAJA Ouput tensors 185: ${outputTensors[185].shape toString()}');
    print('BAJA Ouput tensors 186: ${outputTensors[186].shape toString()}');
    print('BAJA Ouput tensors 187: ${outputTensors[187].shape toString()}');
    print('BAJA Ouput tensors 188: ${outputTensors[188].shape toString()}');
    print('BAJA Ouput tensors 189: ${outputTensors[189].shape toString()}');
    print('BAJA Ouput tensors 190: ${outputTensors[190].shape toString()}');
    print('BAJA Ouput tensors 191: ${outputTensors[191].shape toString()}');
    print('BAJA Ouput tensors 192: ${outputTensors[192].shape toString()}');
    print('BAJA Ouput tensors 193: ${outputTensors[193].shape toString()}');
    print('BAJA Ouput tensors 194: ${outputTensors[194].shape toString()}');
    print('BAJA Ouput tensors 195: ${outputTensors[195].shape toString()}');
    print('BAJA Ouput tensors 196: ${outputTensors[196].shape toString()}');
    print('BAJA Ouput tensors 197: ${outputTensors[197].shape toString()}');
    print('BAJA Ouput tensors 198: ${outputTensors[198].shape toString()}');
    print('BAJA Ouput tensors 199: ${outputTensors[199].shape toString()}');
    print('BAJA Ouput tensors 200: ${outputTensors[200].shape toString()}');
    print('BAJA Ouput tensors 201: ${outputTensors[201].shape toString()}');
    print('BAJA Ouput tensors 202: ${outputTensors[202].shape toString()}');
    print('BAJA Ouput tensors 203: ${outputTensors[203].shape toString()}');
    print('BAJA Ouput tensors 204: ${outputTensors[204].shape toString()}');
    print('BAJA Ouput tensors 205: ${outputTensors[205].shape toString()}');
    print('BAJA Ouput tensors 206: ${outputTensors[206].shape toString()}');
    print('BAJA Ouput tensors 207: ${outputTensors[207].shape toString()}');
    print('BAJA Ouput tensors 208: ${outputTensors[208].shape toString()}');
    print('BAJA Ouput tensors 209: ${outputTensors[209].shape toString()}');
    print('BAJA Ouput tensors 210: ${outputTensors[210].shape toString()}');
    print('BAJA Ouput tensors 211: ${outputTensors[211].shape toString()}');
    print('BAJA Ouput tensors 212: ${outputTensors[212].shape toString()}');
    print('BAJA Ouput tensors 213: ${outputTensors[213].shape toString()}');
    print('BAJA Ouput tensors 214: ${outputTensors[214].shape toString()}');
    print('BAJA Ouput tensors 215: ${outputTensors[215].shape toString()}');
    print('BAJA Ouput tensors 216: ${outputTensors[216].shape toString()}');
    print('BAJA Ouput tensors 217: ${outputTensors[217].shape toString()}');
    print('BAJA Ouput tensors 218: ${outputTensors[218].shape toString()}');
    print('BAJA Ouput tensors 219: ${outputTensors[219].shape toString()}');
    print('BAJA Ouput tensors 220: ${outputTensors[220].shape toString()}');
    print('BAJA Ouput tensors 221: ${outputTensors[221].shape toString()}');
    print('BAJA Ouput tensors 222: ${outputTensors[222].shape toString()}');
    print('BAJA Ouput tensors 223: ${outputTensors[223].shape toString()}');
    print('BAJA Ouput tensors 224: ${outputTensors[224].shape toString()}');
    print('BAJA Ouput tensors 225: ${outputTensors[225].shape toString()}');
    print('BAJA Ouput tensors 226: ${outputTensors[226].shape toString()}');
    print('BAJA Ouput tensors 227: ${outputTensors[227].shape toString()}');
    print('BAJA Ouput tensors 228: ${outputTensors[228].shape toString()}');
    print('BAJA Ouput tensors 229: ${outputTensors[229].shape toString()}');
    print('BAJA Ouput tensors 230: ${outputTensors[230].shape toString()}');
    print('BAJA Ouput tensors 231: ${outputTensors[231].shape toString()}');
    print('BAJA Ouput tensors 232: ${outputTensors[232].shape toString()}');
    print('BAJA Ouput tensors 233: ${outputTensors[233].shape toString()}');
    print('BAJA Ouput tensors 234: ${outputTensors[234].shape toString()}');
    print('BAJA Ouput tensors 235: ${outputTensors[235].shape toString()}');
    print('BAJA Ouput tensors 236: ${outputTensors[236].shape toString()}');
    print('BAJA Ouput tensors 237: ${outputTensors[237].shape toString()}');
    print('BAJA Ouput tensors 238: ${outputTensors[238].shape toString()}');
    print('BAJA Ouput tensors 239: ${outputTensors[239].shape toString()}');
    print('BAJA Ouput tensors 240: ${outputTensors[240].shape toString()}');
    print('BAJA Ouput tensors 241: ${outputTensors[241].shape toString()}');
    print('BAJA Ouput tensors 242: ${outputTensors[242].shape toString()}');
    print('BAJA Ouput tensors 243: ${outputTensors[243].shape toString()}');
    print('BAJA Ouput tensors 244: ${outputTensors[244].shape toString()}');
    print('BAJA Ouput tensors 245: ${outputTensors[245].shape toString()}');
    print('BAJA Ouput tensors 246: ${outputTensors[246].shape toString()}');
    print('BAJA Ouput tensors 247: ${outputTensors[247].shape toString()}');
    print('BAJA Ouput tensors 248: ${outputTensors[248].shape toString()}');
    print('BAJA Ouput tensors 249: ${outputTensors[249].shape toString()}');
    print('BAJA Ouput tensors 250: ${outputTensors[250].shape toString()}');
    print('BAJA Ouput tensors 251: ${outputTensors[251].shape toString()}');
    print('BAJA Ouput tensors 252: ${outputTensors[252].shape toString()}');
    print('BAJA Ouput tensors 253: ${outputTensors[253].shape toString()}');
    print('BAJA Ouput tensors 254: ${outputTensors[254].shape toString()}');
    print('BAJA Ouput tensors 255: ${outputTensors[255].shape toString()}');
    print('BAJA Ouput tensors 256: ${outputTensors[256].shape toString()}');
    print('BAJA Ouput tensors 257: ${outputTensors[257].shape toString()}');
    print('BAJA Ouput tensors 258: ${outputTensors[258].shape toString()}');
    print('BAJA Ouput tensors 259: ${outputTensors[259].shape toString()}');
    print('BAJA Ouput tensors 260: ${outputTensors[260].shape toString()}');
    print('BAJA Ouput tensors 261: ${outputTensors[261].shape toString()}');
    print('BAJA Ouput tensors 262: ${outputTensors[262].shape toString()}');
    print('BAJA Ouput tensors 263: ${outputTensors[263].shape toString()}');
    print('BAJA Ouput tensors 264: ${outputTensors[264].shape toString()}');
    print('BAJA Ouput tensors 265: ${outputTensors[265].shape toString()}');
    print('BAJA Ouput tensors 266: ${outputTensors[266].shape toString()}');
    print('BAJA Ouput tensors 267: ${outputTensors[267].shape toString()}');
    print('BAJA Ouput tensors 268: ${outputTensors[268].shape toString()}');
    print('BAJA Ouput tensors 269: ${outputTensors[269].shape toString()}');
    print('BAJA Ouput tensors 270: ${outputTensors[270].shape toString()}');
    print('BAJA Ouput tensors 271: ${outputTensors[271].shape toString()}');
    print('BAJA Ouput tensors 272: ${outputTensors[272].shape toString()}');
    print('BAJA Ouput tensors 273: ${outputTensors[273].shape toString()}');
    print('BAJA Ouput tensors 274: ${outputTensors[274].shape toString()}');
    print('BAJA Ouput tensors 275: ${outputTensors[275].shape toString()}');
    print('BAJA Ouput tensors 276: ${outputTensors[276].shape toString()}');
    print('BAJA Ouput tensors 277: ${outputTensors[277].shape toString()}');
    print('BAJA Ouput tensors 278: ${outputTensors[278].shape toString()}');
    print('BAJA Ouput tensors 279: ${outputTensors[279].shape toString()}');
    print('BAJA Ouput tensors 280: ${outputTensors[280].shape toString()}');
    print('BAJA Ouput tensors 281: ${outputTensors[281].shape toString()}');
    print('BAJA Ouput tensors 282: ${outputTensors[282].shape toString()}');
    print('BAJA Ouput tensors 283: ${outputTensors[283].shape toString()}');
    print('BAJA Ouput tensors 284: ${outputTensors[284].shape toString()}');
    print('BAJA Ouput tensors 285: ${outputTensors[285].shape toString()}');
    print('BAJA Ouput tensors 286: ${outputTensors[286].shape toString()}');
    print('BAJA Ouput tensors 287: ${outputTensors[287].shape toString()}');
    print('BAJA Ouput tensors 288: ${outputTensors[288].shape toString()}');
    print('BAJA Ouput tensors 289: ${outputTensors[289].shape toString()}');
    print('BAJA Ouput tensors 290: ${outputTensors[290].shape toString()}');
    print('BAJA Ouput tensors 291: ${outputTensors[291].shape toString()}');
    print('BAJA Ouput tensors 292: ${outputTensors[292].shape toString()}');
    print('BAJA Ouput tensors 293: ${outputTensors[293].shape toString()}');
    print('BAJA Ouput tensors 294: ${outputTensors[294].shape toString()}');
    print('BAJA Ouput tensors 295: ${outputTensors[295].shape toString()}');
    print('BAJA Ouput tensors 296: ${outputTensors[296].shape toString()}');
    print('BAJA Ouput tensors 297: ${outputTensors[297].shape toString()}');
    print('BAJA Ouput tensors 298: ${outputTensors[298].shape toString()}');
    print('BAJA Ouput tensors 299: ${outputTensors[299].shape toString()}');
    print('BAJA Ouput tensors 300: ${outputTensors[300].shape toString()}');
    print('BAJA Ouput tensors 301: ${outputTensors[301].shape toString()}');
    print('BAJA Ouput tensors 302: ${outputTensors[302].shape toString()}');
    print('BAJA Ouput tensors 303: ${outputTensors[303].shape toString()}');
    print('BAJA Ouput tensors 304: ${outputTensors[304].shape toString()}');
    print('BAJA Ouput tensors 305: ${outputTensors[305].shape toString()}');
    print('BAJA Ouput tensors 306: ${outputTensors[306].shape toString()}');
    print('BAJA Ouput tensors 307: ${outputTensors[307].shape toString()}');
    print('BAJA Ouput tensors 308: ${outputTensors[308].shape toString()}');
    print('BAJA Ouput tensors 309: ${outputTensors[309].shape toString()}');
    print('BAJA Ouput tensors 310: ${outputTensors[310].shape toString()}');
    print('BAJA Ouput tensors 311: ${outputTensors[311].shape toString()}');
    print('BAJA Ouput tensors 312: ${outputTensors[312].shape toString()}');
    print('BAJA Ouput tensors 313: ${outputTensors[313].shape toString()}');
    print('BAJA Ouput tensors 314: ${outputTensors[314].shape toString()}');
    print('BAJA Ouput tensors 315: ${outputTensors[315].shape toString()}');
    print('BAJA Ouput tensors 316: ${outputTensors[316].shape toString()}');
    print('BAJA Ouput tensors 317: ${outputTensors[317].shape toString()}');
    print('BAJA Ouput tensors 318: ${outputTensors[318].shape toString()}');
    print('BAJA Ouput tensors 319: ${outputTensors[319].shape toString()}');
    print('BAJA Ouput tensors 320: ${outputTensors[320].shape toString()}');
    print('BAJA Ouput tensors 321: ${outputTensors[321].shape toString()}');
    print('BAJA Ouput tensors 322: ${outputTensors[322].shape toString()}');
    print('BAJA Ouput tensors 323: ${outputTensors[323].shape toString()}');
    print('BAJA Ouput tensors 324: ${outputTensors[324].shape toString()}');
    print('BAJA Ouput tensors 325: ${outputTensors[325].shape toString()}');
    print('BAJA Ouput tensors 326: ${outputTensors[326].shape toString()}');
    print('BAJA Ouput tensors 327: ${outputTensors[327].shape toString()}');
    print('BAJA Ouput tensors 328: ${outputTensors[328].shape toString()}');
    print('BAJA Ouput tensors 329: ${outputTensors[329].shape toString()}');
    print('BAJA Ouput tensors 330: ${outputTensors[330].shape toString()}');
    print('BAJA Ouput tensors 331: ${outputTensors[331].shape toString()}');
    print('BAJA Ouput tensors 332: ${outputTensors[332].shape toString()}');
    print('BAJA Ouput tensors 333: ${outputTensors[333].shape toString()}');
    print('BAJA Ouput tensors 334: ${outputTensors[334].shape toString()}');
    print('BAJA Ouput tensors 335: ${outputTensors[335].shape toString()}');
    print('BAJA Ouput tensors 336: ${outputTensors[336].shape toString()}');
    print('BAJA Ouput tensors 337: ${outputTensors[337].shape toString()}');
    print('BAJA Ouput tensors 338: ${outputTensors[338].shape toString()}');
    print('BAJA Ouput tensors 339: ${outputTensors[339].shape toString()}');
    print('BAJA Ouput tensors 340: ${outputTensors[340].shape toString()}');
    print('BAJA Ouput tensors 341: ${outputTensors[341].shape toString()}');
    print('BAJA Ouput tensors 342: ${outputTensors[342].shape toString()}');
    print('BAJA Ouput tensors 343: ${outputTensors[343].shape toString()}');
    print('BAJA Ouput tensors 344: ${outputTensors[344].shape toString()}');
    print('BAJA Ouput tensors 345: ${outputTensors[345].shape toString()}');
    print('BAJA Ouput tensors 346: ${outputTensors[346].shape toString()}');
    print('BAJA Ouput tensors 347: ${outputTensors[347].shape toString()}');
    print('BAJA Ouput tensors 348: ${outputTensors[348].shape toString()}');
    print('BAJA Ouput tensors 349: ${outputTensors[349].shape toString()}');
    print('BAJA Ouput tensors 350: ${outputTensors[350].shape toString()}');
    print('BAJA Ouput tensors 351: ${outputTensors[351].shape toString()}');
    print('BAJA Ouput tensors 352: ${outputTensors[352].shape toString()}');
    print('BAJA Ouput tensors 353: ${outputTensors[353].shape toString()}');
    print('BAJA Ouput tensors 354: ${outputTensors[354].shape toString()}');
    print('BAJA Ouput tensors 355: ${outputTensors[355].shape toString()}');
    print('BAJA Ouput tensors 356: ${outputTensors[356].shape toString()}');
    print('BAJA Ouput tensors 357: ${outputTensors[357].shape toString()}');
    print('BAJA Ouput tensors 358: ${outputTensors[358].shape toString()}');
    print('BAJA Ouput tensors 359: ${outputTensors[359].shape toString()}');
    print('BAJA Ouput tensors 360: ${outputTensors[360].shape toString()}');
    print('BAJA Ouput tensors 361: ${outputTensors[361].shape toString()}');
    print('BAJA Ouput tensors 362: ${outputTensors[362].shape toString()}');
    print('BAJA Ouput tensors 363: ${outputTensors[363].shape toString()}');
    print('BAJA Ouput tensors 364: ${outputTensors[364].shape toString()}');
    print('BAJA Ouput tensors 365: ${outputTensors[365].shape toString()}');
    print('BAJA Ouput tensors 366: ${outputTensors[366].shape toString()}');
    print('BAJA Ouput tensors 367: ${outputTensors[367].shape toString()}');
    print('BAJA Ouput tensors 368: ${outputTensors[368].shape toString()}');
    print('BAJA Ouput tensors 369: ${outputTensors[369].shape toString()}');
    print('BAJA Ouput tensors 370: ${outputTensors[370].shape toString()}');
    print('BAJA Ouput tensors 371: ${outputTensors[371].shape toString()}');
    print('BAJA Ouput tensors 372: ${outputTensors[372].shape toString()}');
    print('BAJA Ouput tensors 373: ${outputTensors[373].shape toString()}');
    print('BAJA Ouput tensors 374: ${outputTensors[374].shape toString()}');
    print('BAJA Ouput tensors 375: ${outputTensors[375].shape toString()}');
    print('BAJA Ouput tensors 376: ${outputTensors[376].shape toString()}');
    print('BAJA Ouput tensors 377: ${outputTensors[377].shape toString()}');
    print('BAJA Ouput tensors 378: ${outputTensors[378].shape toString()}');
    print('BAJA Ouput tensors 379: ${outputTensors[379].shape toString()}');
    print('BAJA Ouput tensors 380: ${outputTensors[380].shape toString()}');
    print('BAJA Ouput tensors 381: ${outputTensors[381].shape toString()}');
    print('BAJA Ouput tensors 382: ${outputTensors[382].shape toString()}');
    print('BAJA Ouput tensors 383: ${outputTensors[383].shape toString()}');
    print('BAJA Ouput tensors 384: ${outputTensors[384].shape toString()}');
    print('BAJA Ouput tensors 385: ${outputTensors[385].shape toString()}');
    print('BAJA Ouput tensors 386: ${outputTensors[386].shape toString()}');
    print('BAJA Ouput tensors 387: ${outputTensors[387].shape toString()}');
    print('BAJA Ouput tensors 388: ${outputTensors[388].shape toString()}');
    print('BAJA Ouput tensors 389: ${outputTensors[389].shape toString()}');
    print('BAJA Ouput tensors 390: ${outputTensors[390].shape toString()}');
    print('BAJA Ouput tensors 391: ${outputTensors[391].shape toString()}');
    print('BAJA Ouput tensors 392: ${outputTensors[392].shape toString()}');
    print('BAJA Ouput tensors 393: ${outputTensors[393].shape toString()}');
    print('BAJA Ouput tensors 394: ${outputTensors[394].shape toString()}');
    print('BAJA Ouput tensors 395: ${outputTensors[395].shape toString()}');
    print('BAJA Ouput tensors 396: ${outputTensors[396].shape toString()}');
    print('BAJA Ouput tensors 397: ${outputTensors[397].shape toString()}');
    print('BAJA Ouput tensors 398: ${outputTensors[398].shape toString()}');
    print('BAJA Ouput tensors 399: ${outputTensors[399].shape toString()}');
    print('BAJA Ouput tensors 400: ${outputTensors[400].shape toString()}');
    print('BAJA Ouput tensors 401: ${outputTensors[401].shape toString()}');
    print('BAJA Ouput tensors 402: ${outputTensors[402].shape toString()}');
    print('BAJA Ouput tensors 403: ${outputTensors[403].shape toString()}');
    print('BAJA Ouput tensors 404: ${outputTensors[404].shape toString()}');
    print('BAJA Ouput tensors 405: ${outputTensors[405].shape toString()}');
    print('BAJA Ouput tensors 406: ${outputTensors[406].shape toString()}');
    print('BAJA Ouput tensors 407: ${outputTensors[407].shape toString()}');
    print('BAJA Ouput tensors 408: ${outputTensors[408].shape toString()}');
    print('BAJA Ouput tensors 409: ${outputTensors[409].shape toString()}');
    print('BAJA Ouput tensors 410: ${outputTensors[410].shape toString()}');
    print('BAJA Ouput tensors 411: ${outputTensors[411].shape toString()}');
    print('BAJA Ouput tensors 412: ${outputTensors[412].shape toString()}');
    print('BAJA Ouput tensors 413: ${outputTensors[413].shape toString()}');
    print('BAJA Ouput tensors 414: ${outputTensors[414].shape toString()}');
    print('BAJA Ouput tensors 415: ${outputTensors[415].shape toString()}');
    print('BAJA Ouput tensors 416: ${outputTensors[416].shape toString()}');
    print('BAJA Ouput tensors 417: ${outputTensors[417].shape toString()}');
    print('BAJA Ouput tensors 418: ${outputTensors[418].shape toString()}');
    print('BAJA Ouput tensors 419: ${outputTensors[419].shape toString()}');
    print('BAJA Ouput tensors 420: ${outputTensors[420].shape toString()}');
    print('BAJA Ouput tensors 421: ${outputTensors[421].shape toString()}');
    print('BAJA Ouput tensors 422: ${outputTensors[422].shape toString()}');
    print('BAJA Ouput tensors 423: ${outputTensors[423].shape toString()}');
    print('BAJA Ouput tensors 424: ${outputTensors[424].shape toString()}');
    print('BAJA Ouput tensors 425: ${outputTensors[425].shape toString()}');
    print('BAJA Ouput tensors 426: ${outputTensors[426].shape toString()}');
    print('BAJA Ouput tensors 427: ${outputTensors[427].shape toString()}');
    print('BAJA Ouput tensors 428: ${outputTensors[428].shape toString()}');
    print('BAJA Ouput tensors 429: ${outputTensors[429].shape toString()}');
    print('BAJA Ouput tensors 430: ${outputTensors[430].shape toString()}');
    print('BAJA Ouput tensors 431: ${outputTensors[431].shape toString()}');
    print('BAJA Ouput tensors 432: ${outputTensors[432].shape toString()}');
    print('BAJA Ouput tensors 433: ${outputTensors[433].shape toString()}');
    print('BAJA Ouput tensors 434: ${outputTensors[434].shape toString()}');
    print('BAJA Ouput tensors 435: ${outputTensors[435].shape toString()}');
    print('BAJA Ouput tensors 436: ${outputTensors[436].shape toString()}');
    print('BAJA Ouput tensors 437: ${outputTensors[437].shape toString()}');
    print('BAJA Ouput tensors 438: ${outputTensors[438].shape toString()}');
    print('BAJA Ouput tensors 439: ${outputTensors[439].shape toString()}');
    print('BAJA Ouput tensors 440: ${outputTensors[440].shape toString()}');
    print('BAJA Ouput tensors 441: ${outputTensors[441].shape toString()}');
    print('BAJA Ouput tensors 442: ${outputTensors[442].shape toString()}');
    print('BAJA Ouput tensors 443: ${outputTensors[443].shape toString()}');
    print('BAJA Ouput tensors 444: ${outputTensors[444].shape toString()}');
    print('BAJA Ouput tensors 445: ${outputTensors[445].shape toString()}');
    print('BAJA Ouput tensors 446: ${outputTensors[446].shape toString()}');
    print('BAJA Ouput tensors 447: ${outputTensors[447].shape toString()}');
    print('BAJA Ouput tensors 448: ${outputTensors[448].shape toString()}');
    print('BAJA Ouput tensors 449: ${outputTensors[449].shape toString()}');
    print('BAJA Ouput tensors 450: ${outputTensors[450].shape toString()}');
    print('BAJA Ouput tensors 451: ${outputTensors[451].shape toString()}');
    print('BAJA Ouput tensors 452: ${outputTensors[452].shape toString()}');
    print('BAJA Ouput tensors 453: ${outputTensors[453].shape toString()}');
    print('BAJA Ouput tensors 454: ${outputTensors[454].shape toString()}');
    print('BAJA Ouput tensors 455: ${outputTensors[455].shape toString()}');
    print('BAJA Ouput tensors 456: ${outputTensors[456].shape toString()}');
    print('BAJA Ouput tensors 457: ${outputTensors[457].shape toString()}');
    print('BAJA Ouput tensors 458: ${outputTensors[458].shape toString()}');
    print('BAJA Ouput tensors 459: ${outputTensors[459].shape toString()}');
    print('BAJA Ouput tensors 460: ${outputTensors[460].shape toString()}');
    print('BAJA Ouput tensors 461: ${outputTensors[461].shape toString()}');
    print('BAJA Ouput tensors 462: ${outputTensors[462].shape toString()}');
    print('BAJA Ouput tensors 463: ${outputTensors[463].shape toString()}');
    print('BAJA Ouput tensors 464: ${outputTensors[464].shape toString()}');
    print('BAJA Ouput tensors 465: ${outputTensors[465].shape toString()}');
    print('BAJA Ouput tensors 466: ${outputTensors[466].shape toString()}');
    print('BAJA Ouput tensors 467: ${outputTensors[467].shape toString()}');
    print('BAJA Ouput tensors 468: ${outputTensors[468].shape toString()}');
    print('BAJA Ouput tensors 469: ${outputTensors[469].shape toString()}');
    print('BAJA Ouput tensors 470: ${outputTensors[470].shape toString()}');
    print('BAJA Ouput tensors 471: ${outputTensors[471].shape toString()}');
    print('BAJA Ouput tensors 472: ${outputTensors[472].shape toString()}');
    print('BAJA Ouput tensors 473: ${outputTensors[473].shape toString()}');
    print('BAJA Ouput tensors 474: ${outputTensors[474].shape toString()}');
    print('BAJA Ouput tensors 475: ${outputTensors[475].shape toString()}');
    print('BAJA Ouput tensors 476: ${outputTensors[476].shape toString()}');
    print('BAJA Ouput tensors 477: ${outputTensors[477].shape toString()}');
    print('BAJA Ouput tensors 478: ${outputTensors[478].shape toString()}');
    print('BAJA Ouput tensors 479: ${outputTensors[479].shape toString()}');
    print('BAJA Ouput tensors 480: ${outputTensors[480].shape toString()}');
    print('BAJA Ouput tensors 481: ${outputTensors[481].shape toString()}');
    print('BAJA Ouput tensors 482: ${outputTensors[482].shape toString()}');
    print('BAJA Ouput tensors 483: ${outputTensors[483].shape toString()}');
    print('BAJA Ouput tensors 484: ${outputTensors[484].shape toString()}');
    print('BAJA Ouput tensors 485: ${outputTensors[485].shape toString()}');
    print('BAJA Ouput tensors 486: ${outputTensors[486].shape toString()}');
    print('BAJA Ouput tensors 487: ${outputTensors[487].shape toString()}');
    print('BAJA Ouput tensors 488: ${outputTensors[488].shape toString()}');
    print('BAJA Ouput tensors 489: ${outputTensors[489].shape toString()}');
    print('BAJA Ouput tensors 490: ${outputTensors[490].shape toString()}');
    print('BAJA Ouput tensors 491: ${outputTensors[491].shape toString()}');
    print('BAJA Ouput tensors 492: ${outputTensors[492].shape toString()}');
    print('BAJA Ouput tensors 493: ${outputTensors[493].shape toString()}');
    print('BAJA Ouput tensors 494: ${outputTensors[494].shape toString()}');
    print('BAJA Ouput tensors 495: ${outputTensors[495].shape toString()}');
    print('BAJA Ouput tensors 496: ${outputTensors[496].shape toString()}');
    print('BAJA Ouput tensors 497: ${outputTensors[497].shape toString()}');
    print('BAJA Ouput tensors 498: ${outputTensors[498].shape toString()}');
    print('BAJA Ouput tensors 499: ${outputTensors[499].shape toString()}');
    print('BAJA Ouput tensors 500: ${outputTensors[500].shape toString()}');
    print('BAJA Ouput tensors 501: ${outputTensors[501].shape toString()}');
    print('BAJA Ouput tensors 502: ${outputTensors[502].shape toString()}');
    print('BAJA Ouput tensors 503: ${outputTensors[503].shape toString()}');
    print('BAJA Ouput tensors 504: ${outputTensors[504].shape toString()}');
    print('BAJA Ouput tensors 505: ${outputTensors[505].shape toString()}');
    print('BAJA Ouput tensors 506: ${outputTensors[506].shape toString()}');
    print('BAJA Ouput tensors 507: ${outputTensors[507].shape toString()}');
    print('BAJA Ouput tensors 508: ${outputTensors[508].shape toString()}');
    print('BAJA Ouput tensors 509: ${outputTensors[509].shape toString()}');
    print('BAJA Ouput tensors 510: ${outputTensors[510].shape toString()}');
    print('BAJA Ouput tensors 511: ${outputTensors[511].shape toString()}');
    print('BAJA Ouput tensors 512: ${outputTensors[512].shape toString()}');
    print('BAJA Ouput tensors 513: ${outputTensors[513].shape toString()}');
    print('BAJA Ouput tensors 514: ${outputTensors[514].shape toString()}');
    print('BAJA Ouput tensors 515: ${outputTensors[515].shape toString()}');
    print('BAJA Ouput tensors 516: ${outputTensors[516].shape toString()}');
    print('BAJA Ouput tensors 517: ${outputTensors[517].shape toString()}');
    print('BAJA Ouput tensors 518: ${outputTensors[518].shape toString()}');
    print('BAJA Ouput tensors 519: ${outputTensors[519].shape toString()}');
    print('BAJA Ouput tensors 520: ${outputTensors[520].shape toString()}');
    print('BAJA Ouput tensors 521: ${outputTensors[521].shape toString()}');
    print('BAJA Ouput tensors 522: ${outputTensors[522].shape toString()}');
    print('BAJA Ouput tensors 523: ${outputTensors[523].shape toString()}');
    print('BAJA Ouput tensors 524: ${outputTensors[524].shape toString()}');
    print('BAJA Ouput tensors 525: ${outputTensors[525].shape toString()}');
    print('BAJA Ouput tensors 526: ${outputTensors[526].shape toString()}');
    print('BAJA Ouput tensors 527: ${outputTensors[527].shape toString()}');
    print('BAJA Ouput tensors 528: ${outputTensors[528].shape toString()}');
    print('BAJA Ouput tensors 529: ${outputTensors[529].shape toString()}');
    print('BAJA Ouput tensors 530: ${outputTensors[530].shape toString()}');
    print('BAJA Ouput tensors 531: ${outputTensors[531].shape toString()}');
    print('BAJA Ouput tensors 532: ${outputTensors[532].shape toString()}');
    print('BAJA Ouput tensors 533: ${outputTensors[533].shape toString()}');
    print('BAJA Ouput tensors 534: ${outputTensors[534].shape toString()}');
    print('BAJA Ouput tensors 535: ${outputTensors[535].shape toString()}');
    print('BAJA Ouput tensors 536: ${outputTensors[536].shape toString()}');
    print('BAJA Ouput tensors 537: ${outputTensors[537].shape toString()}');
    print('BAJA Ouput tensors 538: ${outputTensors[538].shape toString()}');
    print('BAJA Ouput tensors 539: ${outputTensors[539].shape toString()}');
    print('BAJA Ouput tensors 540: ${outputTensors[540].shape toString()}');
    print('BAJA Ouput tensors 541: ${outputTensors[541].shape toString()}');
    print('BAJA Ouput tensors 542: ${outputTensors[542].shape toString()}');
    print('BAJA Ouput tensors 543: ${outputTensors[543].shape toString()}');
    print('BAJA Ouput tensors 544: ${outputTensors[544].shape toString()}');
    print('BAJA Ouput tensors 545: ${outputTensors[545].shape toString()}');
    print('BAJA Ouput tensors 546: ${outputTensors[546].shape toString()}');
    print('BAJA Ouput tensors 547: ${outputTensors[547].shape toString()}');
    print('BAJA Ouput tensors 548: ${outputTensors[548].shape toString()}');
    print('BAJA Ouput tensors 549: ${outputTensors[549].shape toString()}');
    print('BAJA Ouput tensors 550: ${outputTensors[550].shape toString()}');
    print('BAJA Ouput tensors 551: ${outputTensors[551].shape toString()}');
    print('BAJA Ouput tensors 552: ${outputTensors[552].shape toString()}');
    print('BAJA Ouput tensors 553: ${outputTensors[553].shape toString()}');
    print('BAJA Ouput tensors 554: ${outputTensors[554].shape toString()}');
    print('BAJA Ouput tensors 555: ${outputTensors[555].shape toString()}');
    print('BAJA Ouput tensors 556: ${outputTensors[556].shape toString()}');
    print('BAJA Ouput tensors 557: ${outputTensors[557].shape toString()}');
    print('BAJA Ouput tensors 558: ${outputTensors[558].shape toString()}');
    print('BAJA Ouput tensors 559: ${outputTensors[559].shape toString()}');
    print('BAJA Ouput tensors 560: ${outputTensors[560].shape toString()}');
    print('BAJA Ouput tensors 561: ${outputTensors[561].shape toString()}');
    print('BAJA Ouput tensors 562: ${outputTensors[562].shape toString()}');
    print('BAJA Ouput tensors 563: ${outputTensors[563].shape toString()}');
    print('BAJA Ouput tensors 564: ${outputTensors[564].shape toString()}');
    print('BAJA Ouput tensors 565: ${outputTensors[565].shape toString()}');
    print('BAJA Ouput tensors 566: ${outputTensors[566].shape toString()}');
    print('BAJA Ouput tensors 567: ${outputTensors[567].shape toString()}');
    print('BAJA Ouput tensors 568: ${outputTensors[568].shape toString()}');
    print('BAJA Ouput tensors 569: ${outputTensors[569].shape toString()}');
    print('BAJA Ouput tensors 570: ${outputTensors[570].shape toString()}');
    print('BAJA Ouput tensors 571: ${outputTensors[571].shape toString()}');
    print('BAJA Ouput tensors 572: ${outputTensors[572].shape toString()}');
    print('BAJA Ouput tensors 573: ${outputTensors[573].shape toString()}');
    print('BAJA Ouput tensors 574: ${outputTensors[574].shape toString()}');
    print('BAJA Ouput tensors 575: ${outputTensors[575].shape toString()}');
    print('BAJA Ouput tensors 576: ${outputTensors[576].shape toString()}');
    print('BAJA Ouput tensors 577: ${outputTensors[577].shape toString()}');
    print('BAJA Ouput tensors 578: ${outputTensors[578].shape toString()}');
    print('BAJA Ouput tensors 579: ${outputTensors[579].shape toString()}');
    print('BAJA Ouput tensors 580: ${outputTensors[580].shape toString()}');
    print('BAJA Ouput tensors 581: ${outputTensors[581].shape toString()}');
    print('BAJA Ouput tensors 582: ${outputTensors[582].shape toString()}');
    print('BAJA Ouput tensors 583: ${outputTensors[583].shape toString()}');
    print('BAJA Ouput tensors 584: ${outputTensors[584].shape toString()}');
    print('BAJA Ouput tensors 585: ${outputTensors[585].shape toString()}');
    print('BAJA Ouput tensors 586: ${outputTensors[586].shape toString()}');
    print('BAJA Ouput tensors 587: ${outputTensors[587].shape toString()}');
    print('BAJA Ouput tensors 588: ${outputTensors[588].shape toString()}');
    print('BAJA Ouput tensors 589: ${outputTensors[589].shape toString()}');
    print('BAJA Ouput tensors 590: ${outputTensors[590].shape toString()}');
    print('BAJA Ouput tensors 591: ${outputTensors[591].shape toString()}');
    print('BAJA Ouput tensors 592: ${outputTensors[592].shape toString()}');
    print('BAJA Ouput tensors 593: ${outputTensors[593].shape toString()}');
    print('BAJA Ouput tensors 594: ${outputTensors[594].shape toString()}');
    print('BAJA Ouput tensors 595: ${outputTensors[595].shape toString()}');
    print('BAJA Ouput tensors 596: ${outputTensors[596].shape toString()}');
    print('BAJA Ouput tensors 597: ${outputTensors[597].shape toString()}');
    print('BAJA Ouput tensors 598: ${outputTensors[598].shape toString()}');
    print('BAJA Ouput tensors 599: ${outputTensors[599].shape toString()}');
    print('BAJA Ouput tensors 600: ${outputTensors[600].shape toString()}');
    print('BAJA Ouput tensors 601: ${outputTensors[601].shape toString()}');
    print('BAJA Ouput tensors 602: ${outputTensors[602].shape toString()}');
    print('BAJA Ouput tensors 603: ${outputTensors[603].shape toString()}');
    print('BAJA Ouput tensors 604: ${outputTensors[604].shape toString()}');
    print('BAJA Ouput tensors 605: ${outputTensors[605].shape toString()}');
    print('BAJA Ouput tensors 606: ${outputTensors[606].shape toString()}');
    print('BAJA Ouput tensors 607: ${outputTensors[607].shape toString()}');
    print('BAJA Ouput tensors 608: ${outputTensors[608].shape toString()}');
    print('BAJA Ouput tensors 609: ${outputTensors[609].shape toString()}');
    print('BAJA Ouput tensors 610: ${outputTensors[610].shape toString()}');
    print('BAJA Ouput tensors 611: ${outputTensors[611].shape toString()}');
    print('BAJA Ouput tensors 612: ${outputTensors[612].shape toString()}');
    print('BAJA Ouput tensors 613: ${outputTensors[613].shape toString()}');
    print('BAJA Ouput tensors 614: ${outputTensors[614].shape toString()}');
    print('BAJA Ouput tensors 615: ${outputTensors[615].shape toString()}');
    print('BAJA Ouput tensors 616: ${outputTensors[616].shape toString()}');
    print('BAJA Ouput tensors 617: ${outputTensors[617].shape toString()}');
    print('BAJA Ouput tensors 618: ${outputTensors[618].shape toString()}');
    print('BAJA Ouput tensors 619: ${outputTensors[619].shape toString()}');
    print('BAJA Ouput tensors 620: ${outputTensors[620].shape toString()}');
    print('BAJA Ouput tensors 621: ${outputTensors[621].shape toString()}');
    print('BAJA Ouput tensors 622: ${outputTensors[622].shape toString()}');
    print('BAJA Ouput tensors 623: ${outputTensors[623].shape toString()}');
    print('BAJA Ouput tensors 624: ${outputTensors[624].shape toString()}');
    print('BAJA Ouput tensors 625: ${outputTensors[625].shape toString()}');
    print('BAJA Ouput tensors 626: ${outputTensors[626].shape toString()}');
    print('BAJA Ouput tensors 627: ${outputTensors[627].shape toString()}');
    print('BAJA Ouput tensors 628: ${outputTensors[628].shape toString()}');
    print('BAJA Ouput tensors 629: ${outputTensors[629].shape toString()}');
    print('BAJA Ouput tensors 630: ${outputTensors[630].shape toString()}');
    print('BAJA Ouput tensors 631: ${outputTensors[631].shape toString()}');
    print('BAJA Ouput tensors 632: ${outputTensors[632].shape toString()}');
    print('BAJA Ouput tensors 633: ${outputTensors[633].shape toString()}');
    print('BAJA Ouput tensors 634: ${outputTensors[634].shape toString()}');
    print('BAJA Ouput tensors 635: ${outputTensors[635].shape toString()}');
    print('BAJA Ouput tensors 636: ${outputTensors[636].shape toString()}');
    print('BAJA Ouput tensors 637: ${outputTensors[637].shape toString()}');
    print('BAJA Ouput tensors 638: ${outputTensors[638].shape toString()}');
    print('BAJA Ouput tensors 639: ${outputTensors[639].shape toString()}');
    print('BAJA Ouput tensors 640: ${outputTensors[640].shape toString()}');
    print('BAJA Ouput tensors 641: ${outputTensors[641].shape toString()}');
    print('BAJA Ouput tensors 642: ${outputTensors[642].shape toString()}');
    print('BAJA Ouput tensors 643: ${outputTensors[643].shape toString()}');
    print('BAJA Ouput tensors 644: ${outputTensors[644].shape toString()}');
    print('BAJA Ouput tensors 645: ${outputTensors[645].shape toString()}');
    print('BAJA Ouput tensors 646: ${outputTensors[646].shape toString()}');
    print('BAJA Ouput tensors 647: ${outputTensors[647].shape toString()}');
    print('BAJA Ouput tensors 648: ${outputTensors[648].shape toString()}');
    print('BAJA Ouput tensors 649: ${outputTensors[649].shape toString()}');
    print('BAJA Ouput tensors 650: ${outputTensors[650].shape toString()}');
    print('BAJA Ouput tensors 651: ${outputTensors[651].shape toString()}');
    print('BAJA Ouput tensors 652: ${outputTensors[652].shape toString()}');
    print('BAJA Ouput tensors 653: ${outputTensors[653].shape toString()}');
    print('BAJA Ouput tensors 654: ${outputTensors[654].shape toString()}');
    print('BAJA Ouput tensors 655: ${outputTensors[655].shape toString()}');
    print('BAJA Ouput tensors 656: ${outputTensors[656].shape toString()}');
    print('BAJA Ouput tensors 657: ${outputTensors[657].shape toString()}');
    print('BAJA Ouput tensors 658: ${outputTensors[658].shape toString()}');
    print('BAJA Ouput tensors 659: ${outputTensors[659].shape toString()}');
    print('BAJA Ouput tensors 660: ${outputTensors[660].shape toString()}');
    print('BAJA Ouput tensors 661: ${outputTensors[661].shape toString()}');
    print('BAJA Ouput tensors 662: ${outputTensors[662].shape toString()}');
    print('BAJA Ouput tensors 663: ${outputTensors[663].shape toString()}');
    print('BAJA Ouput tensors 664: ${outputTensors[664].shape toString()}');
    print('BAJA Ouput tensors 665: ${outputTensors[665].shape toString()}');
    print('BAJA Ouput tensors 666: ${outputTensors[666].shape toString()}');
    print('BAJA Ouput tensors 667: ${outputTensors[667].shape toString()}');
    print('BAJA Ouput tensors 668: ${outputTensors[668].shape toString()}');
    print('BAJA Ouput tensors 669: ${outputTensors[669].shape toString()}');
    print('BAJA Ouput tensors 670: ${outputTensors[670].shape toString()}');
    print('BAJA Ouput tensors 671: ${outputTensors[671].shape toString()}');
    print('BAJA Ouput tensors 672: ${outputTensors[672].shape toString()}');
    print('BAJA Ouput tensors 673: ${outputTensors[673].shape toString()}');
    print('BAJA Ouput tensors 674: ${outputTensors[674].shape toString()}');
    print('BAJA Ouput tensors 675: ${outputTensors[675].shape toString()}');
    print('BAJA Ouput tensors 676: ${outputTensors[676].shape toString()}');
    print('BAJA Ouput tensors 677: ${outputTensors[677].shape toString()}');
    print('BAJA Ouput tensors 678: ${outputTensors[678].shape toString()}');
    print('BAJA Ouput tensors 679: ${outputTensors[679].shape toString()}');
    print('BAJA Ouput tensors 680: ${outputTensors[680].shape toString()}');
    print('BAJA Ouput tensors 681: ${outputTensors[681].shape toString()}');
    print('BAJA Ouput tensors 682: ${outputTensors[682].shape toString()}');
    print('BAJA Ouput tensors 683: ${outputTensors[683].shape toString()}');
    print('BAJA Ouput tensors 684: ${outputTensors[684].shape toString()}');
    print('BAJA Ouput tensors 685: ${outputTensors[685].shape toString()}');
    print('BAJA Ouput tensors 686: ${outputTensors[686].shape toString()}');
    print('BAJA Ouput tensors 687: ${outputTensors[687].shape toString()}');
    print('BAJA Ouput tensors 688: ${outputTensors[688].shape toString()}');
    print('BAJA Ouput tensors 689: ${outputTensors[689].shape toString()}');
    print('BAJA Ouput tensors 690: ${outputTensors[690].shape toString()}');
    print('BAJA Ouput tensors 691: ${outputTensors[691].shape toString()}');
    print('BAJA Ouput tensors 692: ${outputTensors[692].shape toString()}');
    print('BAJA Ouput tensors 693: ${outputTensors[693].shape toString()}');
    print('BAJA Ouput tensors 694: ${outputTensors[694].shape toString()}');
    print('BAJA Ouput tensors 695: ${outputTensors[695].shape toString()}');
    print('BAJA Ouput tensors 696: ${outputTensors[696].shape toString()}');
    print('BAJA Ouput tensors 697: ${outputTensors[697].shape toString()}');
    print('BAJA Ouput tensors 698: ${outputTensors[698].shape toString()}');
    print('BAJA Ouput tensors 699: ${outputTensors[699].shape toString()}');
    print('BAJA Ouput tensors 700: ${outputTensors[700].shape toString()}');
    print('BAJA Ouput tensors 701: ${outputTensors[701].shape toString()}');
    print('BAJA Ouput tensors 702: ${outputTensors[702].shape toString()}');
    print('BAJA Ouput tensors 703: ${outputTensors[703].shape toString()}');
    print('BAJA Ouput tensors 704: ${outputTensors[704].shape toString()}');
    print('BAJA Ouput tensors 705: ${outputTensors[705].shape toString()}');
    print('BAJA Ouput tensors 706: ${outputTensors[706].shape toString()}');
    print('BAJA Ouput tensors 707: ${outputTensors[707].shape toString()}');
    print('BAJA Ouput tensors 708: ${outputTensors[708].shape toString()}');
    print('BAJA Ouput tensors 709: ${outputTensors[709].shape toString()}');
    print('BAJA Ouput tensors 710: ${outputTensors[710].shape toString()}');
    print('BAJA Ouput tensors 711: ${outputTensors[711].shape toString()}');
    print('BAJA Ouput tensors 712: ${outputTensors[712].shape toString()}');
    print('BAJA Ouput tensors 713: ${outputTensors[713].shape toString()}');
    print('BAJA Ouput tensors 714: ${outputTensors[714].shape toString()}');
    print('BAJA Ouput tensors 715: ${outputTensors[715].shape toString()}');
    print('BAJA Ouput tensors 716: ${outputTensors[716].shape toString()}');
    print('BAJA Ouput tensors 717: ${outputTensors[717].shape toString()}');
    print('BAJA Ouput tensors 718: ${outputTensors[718].shape toString()}');
    print('BAJA Ouput tensors 719: ${outputTensors[719].shape toString()}');
    print('BAJA Ouput tensors 720: ${outputTensors[720].shape toString()}');
    print('BAJA Ouput tensors 721: ${outputTensors[721].shape toString()}');
    print('BAJA Ouput tensors 722: ${outputTensors[722].shape toString()}');
    print('BAJA Ouput tensors 723: ${outputTensors[723].shape toString()}');
    print('BAJA Ouput tensors 724: ${outputTensors[724].shape toString()}');
    print('BAJA Ouput tensors 725: ${outputTensors[725].shape toString()}');
    print('BAJA Ouput tensors 726: ${outputTensors[726].shape toString()}');
    print('BAJA Ouput tensors 727: ${outputTensors[727].shape toString()}');
    print('BAJA Ouput tensors 728: ${outputTensors[728].shape toString()}');
    print('BAJA Ouput tensors 729: ${outputTensors[729].shape toString()}');
    print('BAJA Ouput tensors 730: ${outputTensors[730].shape toString()}');
    print('BAJA Ouput tensors 731: ${outputTensors[731].shape toString()}');
    print('BAJA Ouput tensors 732: ${outputTensors[732].shape toString()}');
    print('BAJA Ouput tensors 733: ${outputTensors[733].shape toString()}');
    print('BAJA Ouput tensors 734: ${outputTensors[734].shape toString()}');
    print('BAJA Ouput tensors 735: ${outputTensors[735].shape toString()}');
    print('BAJA Ouput tensors 736: ${outputTensors[736].shape toString()}');
    print('BAJA Ouput tensors 737: ${outputTensors[737].shape toString()}');
    print('BAJA Ouput tensors 738: ${outputTensors[738].shape toString()}');
    print('BAJA Ouput tensors 739: ${outputTensors[739].shape toString()}');
    print('BAJA Ouput tensors 740: ${outputTensors[740].shape toString()}');
    print('BAJA Ouput tensors 741: ${outputTensors[741].shape toString()}');
    print('BAJA Ouput tensors 742: ${outputTensors[742].shape toString()}');
    print('BAJA Ouput tensors 743: ${outputTensors[743].shape toString()}');
    print('BAJA Ouput tensors 744: ${outputTensors[744].shape toString()}');
    print('BAJA Ouput tensors 745: ${outputTensors[745].shape toString()}');
    print('BAJA Ouput tensors 746: ${outputTensors[746].shape toString()}');
    print('BAJA Ouput tensors 747: ${outputTensors[747].shape toString()}');
    print('BAJA Ouput tensors 748: ${outputTensors[748].shape toString()}');
    print('BAJA Ouput tensors 749: ${outputTensors[749].shape toString()}');
    print('BAJA Ouput tensors 750: ${outputTensors[750].shape toString()}');
    print('BAJA Ouput tensors 751: ${outputTensors[751].shape toString()}');
    print('BAJA Ouput tensors 752: ${outputTensors[752].shape toString()}');
    print('BAJA Ouput tensors 753: ${outputTensors[753].shape toString()}');
    print('BAJA Ouput tensors 754: ${outputTensors[754].shape toString()}');
    print('BAJA Ouput tensors 755: ${outputTensors[755].shape toString()}');
    print('BAJA Ouput tensors 756: ${outputTensors[756].shape toString()}');
    print('BAJA Ouput tensors 757: ${outputTensors[757].shape toString()}');
    print('BAJA Ouput tensors 758: ${outputTensors[758].shape toString()}');
    print('BAJA Ouput tensors 759: ${outputTensors[759].shape toString()}');
    print('BAJA Ouput tensors 760: ${outputTensors[760].shape toString()}');
    print('BAJA Ouput tensors 761: ${outputTensors[761].shape toString()}');
    print('BAJA Ouput tensors 762: ${outputTensors[762].shape toString()}');
    print('BAJA Ouput tensors 763: ${outputTensors[763].shape toString()}');
    print('BAJA Ouput tensors 764: ${outputTensors[764].shape toString()}');
    print('BAJA Ouput tensors 765: ${outputTensors[765].shape toString()}');
    print('BAJA Ouput tensors 766: ${outputTensors[766].shape toString()}');
    print('BAJA Ouput tensors 767: ${outputTensors[767].shape toString()}');
    print('BAJA Ouput tensors 768: ${outputTensors[768].shape toString()}');
    print('BAJA Ouput tensors 769: ${outputTensors[769].shape toString()}');
    print('BAJA Ouput tensors 770: ${outputTensors[770].shape toString()}');
    print('BAJA Ouput tensors 771: ${outputTensors[771].shape toString()}');
    print('BAJA Ouput tensors 772: ${outputTensors[772].shape toString()}');
    print('BAJA Ouput tensors 773: ${outputTensors[773].shape toString()}');
    print('BAJA Ouput tensors 774: ${outputTensors[774].shape toString()}');
    print('BAJA Ouput tensors 775: ${outputTensors[775].shape toString()}');
    print('BAJA Ouput tensors 776: ${outputTensors[776].shape toString()}');
    print('BAJA Ouput tensors 777: ${outputTensors[777].shape toString()}');
    print('BAJA Ouput tensors 778: ${outputTensors[778].shape toString()}');
    print('BAJA Ouput tensors 779: ${outputTensors[779].shape toString()}');
    print('BAJA Ouput tensors 780: ${outputTensors[780].shape toString()}');
    print('BAJA Ouput tensors 781: ${outputTensors[781].shape toString()}');
    print('BAJA Ouput tensors 782: ${outputTensors[782].shape toString()}');
    print('BAJA Ouput tensors 783: ${outputTensors[783].shape toString()}');
    print('BAJA Ouput tensors 784: ${outputTensors[784].shape toString()}');
    print('BAJA Ouput tensors 785: ${outputTensors[785].shape toString()}');
    print('BAJA Ouput tensors 786: ${outputTensors[786].shape toString()}');
    print('BAJA Ouput tensors 787: ${outputTensors[787].shape toString()}');
    print('BAJA Ouput tensors 788: ${outputTensors[788].shape toString()}');
    print('BAJA Ouput tensors 789: ${outputTensors[789].shape toString()}');
    print('BAJA Ouput tensors 790: ${outputTensors[790].shape toString()}');
    print('BAJA Ouput tensors 791: ${outputTensors[791].shape toString()}');
    print('BAJA Ouput tensors 792: ${outputTensors[792].shape toString()}');
    print('BAJA Ouput tensors 793: ${outputTensors[793].shape toString()}');
    print('BAJA Ouput tensors 794: ${outputTensors[794].shape toString()}');
    print('BAJA Ouput tensors 795: ${outputTensors[795].shape toString()}');
    print('BAJA Ouput tensors 796: ${outputTensors[796].shape toString()}');
    print('BAJA Ouput tensors 797: ${outputTensors[797].shape toString()}');
    print('BAJA Ouput tensors 798: ${outputTensors[798].shape toString()}');
    print('BAJA Ouput tensors 799: ${outputTensors[799].shape toString()}');
    print('BAJA Ouput tensors 800: ${outputTensors[800].shape toString()}');
    print('BAJA Ouput tensors 801: ${outputTensors[801].shape toString()}');
    print('BAJA Ouput tensors 802: ${outputTensors[802].shape toString()}');
    print('BAJA Ouput tensors 803: ${outputTensors[803].shape toString()}');
    print('BAJA Ouput tensors 804: ${outputTensors[804].shape toString()}');
    print('BAJA Ouput tensors 805: ${outputTensors[805].shape toString()}');
    print('BAJA Ouput tensors 806: ${outputTensors[806].shape toString()}');
    print('BAJA Ouput tensors 807: ${outputTensors[807].shape toString()}');
    print('BAJA Ouput tensors 808: ${outputTensors[808].shape toString()}');
    print('BAJA Ouput tensors 809: ${outputTensors[809].shape toString()}');
    print('BAJA Ouput tensors 810: ${outputTensors[810].shape toString()}');
    print('BAJA Ouput tensors 811: ${outputTensors[811].shape toString()}');
    print('BAJA Ouput tensors 812: ${outputTensors[812].shape toString()}');
    print('BAJA Ouput tensors 813: ${outputTensors[813].shape toString()}');
    print('BAJA Ouput tensors 814: ${outputTensors[814].shape toString()}');
    print('BAJA Ouput tensors 815: ${outputTensors[815].shape toString()}');
  }

  Future<List<Map<String, dynamic>>> _model2masks(
    Map<int, Object> output,
  ) async {
    // Extract the predictions and prototype masks from model outputs
    final preds0 = output[0] as List<List<List<num>>>;
    final preds1 = output[1] as List<List<List<List<num>>>>;

    // Process the predictions through non-max suppression and mask processing
    return postprocess(preds0, preds1);
  }

  /// A class to help with matrix operations for mask processing
  class Matrix {
    /// Converts a list of lists to a 2D Float32List
    static Float32List listToFloat32List2D(List<List<double>> list) {
      final int height = list.length;
      final int width = list[0].length;
      final Float32List result = Float32List(height * width);
      
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          result[y * width + x] = list[y][x];
        }
      }
      return result;
    }

    /// Converts a 3D list to a 3D Float32List
    static Float32List listToFloat32List3D(List<List<List<double>>> list) {
      final int depth = list.length;
      final int height = list[0].length;
      final int width = list[0][0].length;
      final Float32List result = Float32List(depth * height * width);
      
      for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            result[(z * height + y) * width + x] = list[z][y][x];
          }
        }
      }
      return result;
    }
    
    /// Performs matrix multiplication between two matrices
    static List<List<double>> matmul(List<List<double>> a, List<List<double>> b) {
      int aRows = a.length;
      int aCols = a[0].length;
      int bRows = b.length;
      int bCols = b[0].length;
      
      if (aCols != bRows) {
        throw Exception('Matrix dimensions do not match for multiplication');
      }
      
      List<List<double>> result = List.generate(
        aRows, (_) => List<double>.filled(bCols, 0.0)
      );
      
      for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
          double sum = 0.0;
          for (int k = 0; k < aCols; k++) {
            sum += a[i][k] * b[k][j];
          }
          result[i][j] = sum;
        }
      }
      
      return result;
    }
    
    /// Flattens a 3D tensor to a 2D matrix
    static List<List<double>> reshape3DTo2D(List<List<List<double>>> tensor, {bool keepFirstDim = true}) {
      int dim1 = tensor.length;
      int dim2 = tensor[0].length;
      int dim3 = tensor[0][0].length;
      
      if (keepFirstDim) {
        // Keep first dimension, flatten second and third
        List<List<double>> result = List.generate(
          dim1, (_) => List<double>.filled(dim2 * dim3, 0.0)
        );
        
        for (int i = 0; i < dim1; i++) {
          for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
              result[i][j * dim3 + k] = tensor[i][j][k];
            }
          }
        }
        return result;
      } else {
        // Flatten all dimensions to 2D
        List<List<double>> result = List.generate(
          1, (_) => List<double>.filled(dim1 * dim2 * dim3, 0.0)
        );
        
        for (int i = 0; i < dim1; i++) {
          for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
              result[0][(i * dim2 + j) * dim3 + k] = tensor[i][j][k];
            }
          }
        }
        return result;
      }
    }
    
    /// Reshapes a 2D matrix to a 3D tensor
    static List<List<List<double>>> reshape2DTo3D(List<List<double>> matrix, int dim1, int dim2, int dim3) {
      List<List<List<double>>> result = List.generate(
        dim1, (_) => List.generate(
          dim2, (_) => List<double>.filled(dim3, 0.0)
        )
      );
      
      for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
          for (int k = 0; k < dim3; k++) {
            result[i][j][k] = matrix[i][j * dim3 + k];
          }
        }
      }
      
      return result;
    }
    
    /// Creates a range of integers as a 1D list
    static List<int> arange(int start, int stop) {
      List<int> result = List<int>.filled(stop - start, 0);
      for (int i = 0; i < stop - start; i++) {
        result[i] = start + i;
      }
      return result;
    }
    
    /// Maximum of two values
    static double max(double a, double b) {
      return a > b ? a : b;
    }
    
    /// Minimum of two values
    static double min(double a, double b) {
      return a < b ? a : b;
    }
    
    /// Creates an array of ones with the specified shape
    static List<List<double>> ones(int rows, int cols) {
      return List.generate(rows, (_) => List<double>.filled(cols, 1.0));
    }
    
    /// Creates a meshgrid from x and y arrays
    static List<List<List<int>>> meshgrid(List<int> x, List<int> y) {
      List<List<int>> xx = List.generate(
        y.length, (_) => List<int>.from(x)
      );
      
      List<List<int>> yy = List.generate(
        y.length, (i) => List<int>.filled(x.length, y[i])
      );
      
      return [xx, yy];
    }
    
    /// Stacks a list of 2D arrays along a new axis
    static List<List<List<double>>> stack(List<List<List<double>>> arrays) {
      int depth = arrays.length;
      int height = arrays[0].length;
      int width = arrays[0][0].length;
      
      List<List<List<double>>> result = List.generate(
        depth, (i) => List.generate(
          height, (j) => List<double>.from(arrays[i][j])
        )
      );
      
      return result;
    }
  }

  /// Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format
  /// where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
  List<List<double>> xywh2xyxy(List<List<double>> boxes) {
    List<List<double>> result = [];
    for (int i = 0; i < boxes.length; i++) {
      List<double> box = List<double>.from(boxes[i]); // Make a copy
      double centerX = boxes[i][0];
      double centerY = boxes[i][1];
      double width = boxes[i][2];
      double height = boxes[i][3];
      
      box[0] = centerX - width / 2; // top left x
      box[1] = centerY - height / 2; // top left y
      box[2] = centerX + width / 2; // bottom right x
      box[3] = centerY + height / 2; // bottom right y
      
      result.add(box);
    }
    return result;
  }

  /// Non-maximum suppression (NMS) to filter out overlapping bounding boxes
  List<int> nms(List<List<double>> boxes, List<double> scores, double iouThreshold) {
    // If no boxes, return empty list
    if (boxes.isEmpty) return [];
    
    // Get coordinates of bounding boxes
    List<double> x1 = boxes.map((box) => box[0]).toList();
    List<double> y1 = boxes.map((box) => box[1]).toList();
    List<double> x2 = boxes.map((box) => box[2]).toList();
    List<double> y2 = boxes.map((box) => box[3]).toList();
    
    // Calculate areas of bounding boxes
    List<double> areas = [];
    for (int i = 0; i < boxes.length; i++) {
      areas.add((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
    }
    
    // Sort by scores in descending order
    List<int> order = List.generate(scores.length, (i) => i);
    order.sort((a, b) => scores[b].compareTo(scores[a]));
    
    List<int> keep = [];
    
    // While there are boxes to process
    while (order.isNotEmpty) {
      int i = order[0]; // Get box with highest score
      keep.add(i); // Add to selected boxes
      
      if (order.length == 1) break; // If only one box left, break
      
      // Calculate IoU with remaining boxes
      List<double> xx1 = [];
      List<double> yy1 = [];
      List<double> xx2 = [];
      List<double> yy2 = [];
      
      for (int j = 1; j < order.length; j++) {
        int idx = order[j];
        xx1.add(Matrix.max(x1[i], x1[idx]));
        yy1.add(Matrix.max(y1[i], y1[idx]));
        xx2.add(Matrix.min(x2[i], x2[idx]));
        yy2.add(Matrix.min(y2[i], y2[idx]));
      }
      
      // Calculate width and height of intersection
      List<double> w = [];
      List<double> h = [];
      for (int j = 0; j < xx1.length; j++) {
        w.add(Matrix.max(0.0, xx2[j] - xx1[j] + 1));
        h.add(Matrix.max(0.0, yy2[j] - yy1[j] + 1));
      }
      
      // Calculate area of intersection
      List<double> inter = [];
      for (int j = 0; j < w.length; j++) {
        inter.add(w[j] * h[j]);
      }
      
      // Calculate IoU
      List<double> iou = [];
      for (int j = 0; j < inter.length; j++) {
        int idx = order[j + 1]; // +1 because we skip the first element (i)
        iou.add(inter[j] / (areas[i] + areas[idx] - inter[j]));
      }
      
      // Find indices of boxes with IoU less than threshold
      List<int> inds = [];
      for (int j = 0; j < iou.length; j++) {
        if (iou[j] <= iouThreshold) {
          inds.add(j + 1); // +1 because we skip the first element (i)
        }
      }
      
      // Update order to include only remaining boxes
      List<int> newOrder = [];
      for (int idx in inds) {
        newOrder.add(order[idx]);
      }
      order = newOrder;
    }
    
    return keep;
  }

  /// Perform non-maximum suppression on a set of boxes, with support for masks
  List<List<Map<String, dynamic>>> nonMaxSuppression(
    List<List<List<num>>> prediction, {
    double confThres = 0.25,
    double iouThres = 0.9,
    List<int>? classes,
    bool agnostic = false,
    int maxDet = 32,
    int nc = 0,
    double maxTimeImg = 0.05,
    int maxNms = 30000,
    int maxWh = 7680,
  }) {
    int bs = prediction.length; // batch size
    nc = nc > 0 ? nc : prediction[0].length - 4; // number of classes
    int nm = prediction[0][0].length - nc - 4; // number of masks
    int mi = 4 + nc; // mask start index
    
    // Convert prediction format and extract candidates
    List<List<bool>> xc = [];
    for (int i = 0; i < prediction.length; i++) {
      List<bool> xcRow = [];
      for (int j = 0; j < prediction[i].length; j++) {
        double maxConf = 0.0;
        for (int k = 4; k < 4 + nc; k++) {
          double val = prediction[i][j][k].toDouble();
          if (val > maxConf) maxConf = val;
        }
        xcRow.add(maxConf > confThres);
      }
      xc.add(xcRow);
    }
    
    // Initialize output
    List<List<Map<String, dynamic>>> output = List.generate(
      bs, (_) => []
    );
    
    // For each batch
    for (int xi = 0; xi < prediction.length; xi++) {
      // Get candidates
      List<List<double>> x = [];
      for (int j = 0; j < prediction[xi].length; j++) {
        if (xc[xi][j]) {
          List<double> row = prediction[xi][j].map((val) => val.toDouble()).toList();
          x.add(row);
        }
      }
      
      if (x.isEmpty) continue; // Skip if no candidates
      
      // Extract boxes, classes and masks
      List<List<double>> box = [];
      List<List<double>> cls = [];
      List<List<double>> mask = [];
      
      for (int j = 0; j < x.length; j++) {
        box.add(x[j].sublist(0, 4));
        cls.add(x[j].sublist(4, 4 + nc));
        mask.add(x[j].sublist(4 + nc, 4 + nc + nm));
      }
      
      // Convert (center_x, center_y, width, height) to (x1, y1, x2, y2)
      box = xywh2xyxy(box);
      
      // Find max confidence and class index for each detection
      List<double> conf = [];
      List<int> j = [];
      
      for (int i = 0; i < cls.length; i++) {
        double maxVal = 0.0;
        int maxIdx = 0;
        for (int k = 0; k < cls[i].length; k++) {
          if (cls[i][k] > maxVal) {
            maxVal = cls[i][k];
            maxIdx = k;
          }
        }
        conf.add(maxVal);
        j.add(maxIdx);
      }
      
      // Combine information and filter by confidence
      List<Map<String, dynamic>> detections = [];
      for (int i = 0; i < box.length; i++) {
        if (conf[i] > confThres) {
          Map<String, dynamic> detection = {
            'box': box[i],
            'conf': conf[i],
            'cls': j[i].toDouble(),
            'mask': mask[i],
          };
          detections.add(detection);
        }
      }
      
      // Sort by confidence
      detections.sort((a, b) => b['conf'].compareTo(a['conf']));
      if (detections.length > maxNms) {
        detections = detections.sublist(0, maxNms);
      }
      
      // Apply NMS
      List<List<double>> boxes = detections.map((d) => d['box'] as List<double>).toList();
      List<double> scores = detections.map((d) => d['conf'] as double).toList();
      
      // Adjust boxes for batched NMS if not agnostic
      if (!agnostic) {
        for (int i = 0; i < boxes.length; i++) {
          boxes[i][0] += (detections[i]['cls'] as double) * maxWh;
          boxes[i][1] += (detections[i]['cls'] as double) * maxWh;
          boxes[i][2] += (detections[i]['cls'] as double) * maxWh;
          boxes[i][3] += (detections[i]['cls'] as double) * maxWh;
        }
      }
      
      List<int> keep = nms(boxes, scores, iouThres);
      
      // Limit detections
      if (keep.length > maxDet) keep = keep.sublist(0, maxDet);
      
      // Create final output for this batch
      List<Map<String, dynamic>> batchOutput = [];
      for (int idx in keep) {
        batchOutput.add(detections[idx]);
      }
      
      output[xi] = batchOutput;
    }
    
    return output;
  }

  /// Crop mask based on bounding box
  List<List<List<bool>>> cropMask(List<List<List<double>>> masks, List<List<double>> boxes) {
    int n = masks.length;
    int h = masks[0].length;
    int w = masks[0][0].length;
    
    // Create coordinate grids
    List<int> rows = Matrix.arange(0, w);
    List<int> cols = Matrix.arange(0, h);
    
    // Create result mask
    List<List<List<bool>>> result = List.generate(
      n, (_) => List.generate(
        h, (_) => List<bool>.filled(w, false)
      )
    );
    
    // For each mask and box
    for (int i = 0; i < n; i++) {
      double x1 = boxes[i][0];
      double y1 = boxes[i][1];
      double x2 = boxes[i][2];
      double y2 = boxes[i][3];
      
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          // Only keep pixels inside the bounding box
          if (x >= x1 && x < x2 && y >= y1 && y < y2) {
            result[i][y][x] = masks[i][y][x] > 0.5; // Convert to binary mask
          }
        }
      }
    }
    
    return result;
  }

  /// Resize a mask to a new shape
  List<List<double>> resizeWithImage(List<List<double>> mask, int newHeight, int newWidth) {
    int originalHeight = mask.length;
    int originalWidth = mask[0].length;
    
    List<List<double>> resized = List.generate(
      newHeight, (_) => List<double>.filled(newWidth, 0.0)
    );
    
    double scaleY = originalHeight / newHeight;
    double scaleX = originalWidth / newWidth;
    
    // Simple nearest neighbor interpolation
    for (int y = 0; y < newHeight; y++) {
      for (int x = 0; x < newWidth; x++) {
        int srcY = (y * scaleY).floor();
        int srcX = (x * scaleX).floor();
        
        srcY = min(srcY, originalHeight - 1);
        srcX = min(srcX, originalWidth - 1);
        
        resized[y][x] = mask[srcY][srcX];
      }
    }
    
    return resized;
  }

  /// Process the mask output from the mask head
  List<List<List<bool>>> processMaskNative(
    List<List<List<double>>> protos,
    List<List<double>> masksIn,
    List<List<double>> bboxes,
    List<int> shape,
  ) {
    int c = protos.length; // number of channels
    int mh = protos[0].length; // mask height
    int mw = protos[0][0].length; // mask width
    
    // Flatten protos: [c, mh, mw] -> [c, mh*mw]
    List<List<double>> protosFlattened = List.generate(c, (i) {
      List<double> flattened = [];
      for (int y = 0; y < mh; y++) {
        for (int x = 0; x < mw; x++) {
          flattened.add(protos[i][y][x]);
        }
      }
      return flattened;
    });
    
    // Matrix multiplication between masks_in and flattened protos
    // masksIn shape: [n, c], protosFlattened shape: [c, mh*mw]
    // Result shape: [n, mh*mw]
    List<List<double>> masks = Matrix.matmul(masksIn, protosFlattened);
    
    // Reshape the result to [n, mh, mw]
    List<List<List<double>>> masksReshaped = [];
    for (int i = 0; i < masks.length; i++) {
      List<List<double>> mask2D = [];
      for (int y = 0; y < mh; y++) {
        List<double> row = [];
        for (int x = 0; x < mw; x++) {
          row.add(masks[i][y * mw + x]);
        }
        mask2D.add(row);
      }
      masksReshaped.add(mask2D);
    }
    
    // Calculate padding and crop masks
    double gain = min(mh / shape[0], mw / shape[1]);
    List<double> pad = [
      (mw - shape[1] * gain) / 2,
      (mh - shape[0] * gain) / 2
    ];
    
    int top = pad[1].floor();
    int left = pad[0].floor();
    int bottom = (mh - pad[1]).floor();
    int right = (mw - pad[0]).floor();
    
    List<List<List<double>>> masksCropped = [];
    for (int i = 0; i < masksReshaped.length; i++) {
      List<List<double>> croppedMask = [];
      for (int y = top; y < bottom; y++) {
        List<double> row = [];
        for (int x = left; x < right; x++) {
          row.add(masksReshaped[i][y][x]);
        }
        croppedMask.add(row);
      }
      masksCropped.add(croppedMask);
    }
    
    // Resize masks to original image shape
    List<List<List<double>>> resizedMasks = [];
    for (int i = 0; i < masksCropped.length; i++) {
      List<List<double>> resizedMask = resizeWithImage(
        masksCropped[i],
        shape[0],
        shape[1]
      );
      resizedMasks.add(resizedMask);
    }
    
    // Apply bounding box cropping
    List<List<List<bool>>> finalMasks = cropMask(resizedMasks, bboxes);
    
    return finalMasks;
  }

  /// Process the model output to extract masks and bounding boxes
  List<Map<String, dynamic>> postprocess(
    List<List<List<num>>> preds0,
    List<List<List<List<num>>>> preds1,
  ) {
    // Convert the predictions to double format
    List<List<List<double>>> predictions = preds0.map((batch) {
      return batch.map((row) {
        return row.map((val) => val.toDouble()).toList();
      }).toList();
    }).toList();
    
    List<List<List<List<double>>>> protos = preds1.map((batch) {
      return batch.map((feature) {
        return feature.map((row) {
          return row.map((val) => val.toDouble()).toList();
        }).toList();
      }).toList();
    }).toList();
    
    // Apply non-maximum suppression
    List<List<Map<String, dynamic>>> pred = nonMaxSuppression(
      predictions,
      confThres: 0.4,
      iouThres: 0.9,
      agnostic: false,
      maxDet: 32,
      nc: 1
    );
    
    // If no objects detected, return empty list
    if (pred.isEmpty || pred[0].isEmpty) {
      return [];
    }
    
    // Extract mask coefficients and bounding boxes
    List<List<double>> maskCoefficients = [];
    List<List<double>> boxes = [];
    
    for (var detection in pred[0]) {
      maskCoefficients.add(detection['mask'] as List<double>);
      boxes.add(detection['box'] as List<double>);
    }
    
    // Process masks
    List<List<List<bool>>> masks = processMaskNative(
      protos[0],
      maskCoefficients,
      boxes,
      [640, 640] // Original image shape
    );
    
    // Create final output with mask and bounding box information
    List<Map<String, dynamic>> result = [];
    for (int i = 0; i < boxes.length; i++) {
      Map<String, dynamic> item = {
        'box': boxes[i],
        'score': pred[0][i]['conf'] as double,
        'class': pred[0][i]['cls'] as double,
        'mask': masks[i],
      };
      result.add(item);
    }
    
    return result;
  }
}
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/prediction_result.dart';

class ApiService {
  static const String _prefKeyBaseUrl = 'kicksense_backend_url';
  
  // Default URL: 10.0.2.2 is Android emulator's alias for host PC localhost.
  // On Windows desktop or web, 127.0.0.1 works directly.
  static const String _defaultUrl = 'http://10.0.2.2:8000';

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefKeyBaseUrl) ?? _defaultUrl;
  }

  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    String cleanUrl = url.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    }
    await prefs.setString(_prefKeyBaseUrl, cleanUrl);
  }

  static Future<bool> checkServerHealth() async {
    try {
      final base = await getBaseUrl();
      final uri = Uri.parse('$base/health');
      final res = await http.get(uri).timeout(const Duration(seconds: 4));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<List<Map<String, dynamic>>> getSampleClips() async {
    try {
      final base = await getBaseUrl();
      final uri = Uri.parse('$base/api/sample-clips');
      final res = await http.get(uri).timeout(const Duration(seconds: 4));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final list = data['samples'] as List<dynamic>? ?? [];
        return list.map((e) => Map<String, dynamic>.from(e as Map)).toList();
      }
    } catch (_) {}
    return [];
  }

  static Future<PredictionResult> predictSample(String sampleId) async {
    final base = await getBaseUrl();
    final uri = Uri.parse('$base/api/predict-sample?sample_id=$sampleId');
    final res = await http.post(uri).timeout(const Duration(seconds: 25));

    if (res.statusCode == 200) {
      final json = jsonDecode(res.body);
      return PredictionResult.fromJson(json);
    } else {
      String msg = 'Failed to analyze sample';
      try {
        final err = jsonDecode(res.body);
        msg = err['detail'] ?? msg;
      } catch (_) {}
      throw Exception(msg);
    }
  }

  static Future<PredictionResult> predictVideo(File videoFile) async {
    final base = await getBaseUrl();
    final uri = Uri.parse('$base/predict');

    final request = http.MultipartRequest('POST', uri);
    final stream = http.ByteStream(videoFile.openRead());
    final length = await videoFile.length();

    final multipartFile = http.MultipartFile(
      'file',
      stream,
      length,
      filename: videoFile.path.split(Platform.pathSeparator).last,
    );
    request.files.add(multipartFile);

    final streamedResponse = await request.send().timeout(const Duration(seconds: 45));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final json = jsonDecode(response.body);
      return PredictionResult.fromJson(json);
    } else {
      String msg = 'Analysis failed (${response.statusCode})';
      try {
        final err = jsonDecode(response.body);
        msg = err['detail'] ?? msg;
      } catch (_) {}
      throw Exception(msg);
    }
  }
}

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/prediction_result.dart';

class ApiService {
  static const String _prefKeyBaseUrl = 'kicksense_backend_url';
  
  // Default URL: With ADB reverse, 127.0.0.1:8000 routes directly over USB.
  static const String _defaultUrl = 'http://127.0.0.1:8000';
  // Wi-Fi LAN Fallback: If ADB reverse drops, route directly over local Wi-Fi.
  static const String _lanFallbackUrl = 'http://192.168.100.18:8000';

  static String? _activeUrl;

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final custom = prefs.getString(_prefKeyBaseUrl);
    if (custom != null && custom.isNotEmpty) return custom;

    if (_activeUrl != null) return _activeUrl!;

    // Quick test if 127.0.0.1:8000 (USB adb reverse) is reachable
    try {
      final res = await http.get(Uri.parse('$_defaultUrl/health')).timeout(const Duration(milliseconds: 900));
      if (res.statusCode == 200) {
        _activeUrl = _defaultUrl;
        return _defaultUrl;
      }
    } catch (_) {}

    // Fallback to Wi-Fi LAN IP
    try {
      final res = await http.get(Uri.parse('$_lanFallbackUrl/health')).timeout(const Duration(milliseconds: 1500));
      if (res.statusCode == 200) {
        _activeUrl = _lanFallbackUrl;
        return _lanFallbackUrl;
      }
    } catch (_) {}

    return _defaultUrl;
  }

  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    String cleanUrl = url.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    }
    _activeUrl = cleanUrl;
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

  static const List<Map<String, dynamic>> defaultSamples = [
    {"id": "penalty_06", "name": "World Cup Final — Paredes", "true_direction": "Left", "true_power": 8.0},
    {"id": "penalty_07", "name": "World Cup Final — Kolo Muani", "true_direction": "Right", "true_power": 8.0},
    {"id": "penalty_08", "name": "World Cup Final — Montiel (Match Winner)", "true_direction": "Right", "true_power": 10.0},
    {"id": "penalty_10", "name": "World Cup Shootout — Messi", "true_direction": "Right", "true_power": 9.0},
    {"id": "penalty_11", "name": "World Cup Shootout — Berghuis", "true_direction": "Right", "true_power": 9.0},
    {"id": "penalty_13", "name": "World Cup Shootout — Koopmeiners", "true_direction": "Left", "true_power": 8.5},
    {"id": "penalty_291", "name": "Recorded Session — Drive", "true_direction": "Left", "true_power": 9.5},
    {"id": "penalty_292", "name": "Recorded Session — Corner", "true_direction": "Right", "true_power": 9.0}
  ];

  static Future<List<Map<String, dynamic>>> getSampleClips() async {
    final urlsToTry = <String>[_defaultUrl, _lanFallbackUrl];
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/api/sample-clips');
        final res = await http.get(uri).timeout(const Duration(seconds: 3));
        if (res.statusCode == 200) {
          _activeUrl = base;
          final data = jsonDecode(res.body);
          final list = data['samples'] as List<dynamic>? ?? [];
          if (list.isNotEmpty) {
            return list.map((e) => Map<String, dynamic>.from(e as Map)).toList();
          }
        }
      } catch (_) {}
    }
    return defaultSamples;
  }

  static Future<PredictionResult> predictSample(String sampleId) async {
    final urlsToTry = <String>[];
    final currentBase = await getBaseUrl();
    urlsToTry.add(currentBase);
    if (!urlsToTry.contains(_lanFallbackUrl)) urlsToTry.add(_lanFallbackUrl);
    if (!urlsToTry.contains(_defaultUrl)) urlsToTry.add(_defaultUrl);

    Exception? lastError;
    for (final base in urlsToTry) {
      try {
        final uri = Uri.parse('$base/api/predict-sample?sample_id=$sampleId');
        final res = await http.post(uri).timeout(const Duration(seconds: 30));

        if (res.statusCode == 200) {
          _activeUrl = base;
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
      } catch (e) {
        lastError = e is Exception ? e : Exception(e.toString());
      }
    }
    throw lastError ?? Exception('Connection error. Please ensure backend is running.');
  }

  static Future<PredictionResult> predictVideo(File videoFile) async {
    final urlsToTry = <String>[];
    final currentBase = await getBaseUrl();
    urlsToTry.add(currentBase);
    if (!urlsToTry.contains(_lanFallbackUrl)) urlsToTry.add(_lanFallbackUrl);
    if (!urlsToTry.contains(_defaultUrl)) urlsToTry.add(_defaultUrl);

    Exception? lastError;
    for (final base in urlsToTry) {
      try {
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
          _activeUrl = base;
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
      } catch (e) {
        lastError = e is Exception ? e : Exception(e.toString());
      }
    }
    throw lastError ?? Exception('Upload failed. Please check server connection.');
  }
}

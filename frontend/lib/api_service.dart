import 'dart:convert';
import 'package:http/http.dart' as http;
import 'constants.dart';

class ApiService {
  static final _client = http.Client();

  /// 공개 디바이스 목록
  /// /devices 응답 예:
  /// {
  ///   "espcam1": {
  ///     "ip": "192.168.x.x",
  ///     "status": "offline" | "online",
  ///     "last_seen": "ISO8601",
  ///     "last_offline_at": "ISO8601"
  ///   }
  /// }
  static Future<Map<String, dynamic>> fetchDevices() async {
    final uri = Uri.parse('$backendBaseUrl/devices');
    final res = await _client.get(uri);
    if (res.statusCode == 200) {
      final data = jsonDecode(res.body);
      if (data is Map<String, dynamic>) return data;
      throw Exception('Unexpected /devices payload shape');
    }
    throw Exception('Failed to load devices: ${res.statusCode}');
  }

  /// flame 값(0/1/-1)
  static Future<int> fetchFlame() async {
    final uri = Uri.parse('$backendBaseUrl/flame/espcam1');
    final res = await _client.get(uri);
    if (res.statusCode == 200) {
      return (jsonDecode(res.body)['flame'] as num).toInt();
    }
    return -1;
  }

  /// MJPEG 스트림 URL (단순 문자열 반환)
        static String snapshotUrl([String? device]) =>
          device == null ? '$backendBaseUrl/snapshot' : '$backendBaseUrl/snapshot/$device';

      static Future<Map<String, dynamic>> startAiStream() async {
        final uri = Uri.parse('$backendBaseUrl/start_ai_stream');
        final res = await _client.post(uri);
        if (res.statusCode == 200) {
          return jsonDecode(res.body) as Map<String, dynamic>;
        }
        throw Exception('Failed to start AI stream: ${res.statusCode} ${res.body}');
      }

      static Future<Map<String, dynamic>> stopAiStream() async {
        final uri = Uri.parse('$backendBaseUrl/stop_ai_stream');
        final res = await _client.post(uri);
        if (res.statusCode == 200) {
          return jsonDecode(res.body) as Map<String, dynamic>;
        }
        throw Exception('Failed to stop AI stream: ${res.statusCode} ${res.body}');
      }

      static Future<Map<String, dynamic>> getAiStreamStatus() async {
        final uri = Uri.parse('$backendBaseUrl/ai_stream_status');
        final res = await _client.get(uri);
        if (res.statusCode == 200) {
          return jsonDecode(res.body) as Map<String, dynamic>;
        }
        throw Exception('Failed to get AI stream status: ${res.statusCode} ${res.body}');
      }
    }
}

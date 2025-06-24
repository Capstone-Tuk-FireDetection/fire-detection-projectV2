import 'dart:convert';
import 'package:http/http.dart' as http;
import 'constants.dart';

class ApiService {
  static final _client = http.Client();

  /// 공개 디바이스 목록
  static Future<Map<String, String>> fetchDevices() async {
    final uri = Uri.parse('$backendBaseUrl/devices');
    final res = await _client.get(uri);
    if (res.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(res.body);
      return data.map((k, v) => MapEntry(k, v.toString()));
    }
    throw Exception('Failed to load devices');
  }

  /// flame 값(0/1/-1)
  static Future<int> fetchFlame() async {
    final uri = Uri.parse('$backendBaseUrl/flame/espcam1');
    final res = await _client.get(uri);
    if (res.statusCode == 200) {
      return jsonDecode(res.body)['flame'] as int;
    }
    return -1;
  }

  /// MJPEG 스트림 URL (단순 문자열 반환)
  static String streamUrl([String? device]) =>
      device == null ? '$backendBaseUrl/stream' : '$backendBaseUrl/stream/$device';
}

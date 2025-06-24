// video_page.dart  ─ 변경·추가된 부분에 ★ 표시
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'api_service.dart';
import 'web_mjpeg_view.dart';
import 'package:flutter_mjpeg/flutter_mjpeg.dart';

class VideoPage extends StatefulWidget {
  const VideoPage({super.key});
  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  late Future<Map<String, String>> _devicesF;
  String? _selected;

  bool _streaming = false;                    // ★ 스트리밍 ON/OFF 상태
  late String _streamUrl;                     // ★ 현재 URL

  Future<int>? _flameF;

  @override
  void initState() {
    super.initState();
    _devicesF = ApiService.fetchDevices();
  }

  void _toggleStream() {                      // ★ 버튼 콜백
    setState(() {
      _streaming = !_streaming;
      if (_streaming) {
        _streamUrl = ApiService.streamUrl(_selected);
        _flameF = ApiService.fetchFlame();
      } else {
        _flameF = null;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, String>>(
      future: _devicesF,
      builder: (ctx, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snap.hasError || snap.data!.isEmpty) {
          return Center(child: Text('장치가 없습니다: ${snap.error ?? ''}'));
        }

        final devices = snap.data!.keys.toList()..sort();
        _selected ??= devices.first;

        return Column(
          children: [
            // ───── 장치 선택 ─────
            Padding(
              padding: const EdgeInsets.all(8),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selected,
                  items: devices
                      .map((d) => DropdownMenuItem(
                            value: d,
                            child: Row(
                              children: [
                                const Icon(Icons.videocam, size: 16),
                                const SizedBox(width: 8),
                                Text(d),
                              ],
                            ),
                          ))
                      .toList(),
                  onChanged: (val) {
                    setState(() {
                      _selected = val;
                      _streaming = false;      // ★ 장치 바꾸면 스트림 OFF
                      _flameF = null;
                    });
                  },
                ),
              ),
            ),

            // ───── 스트림 화면 / 대기화면 ─────
            Expanded(
              child: _streaming
                  ? (kIsWeb
                      ? WebMjpegView(streamUrl: _streamUrl)
                      : Mjpeg(
                          stream: _streamUrl,
                          isLive: true,
                          error: (_, err, __) =>
                              Center(child: Text('스트림 오류: $err')),
                        ))
                  : Center(                    // ★ 스트림 OFF 상태
                      child: Text(
                        '스트림이 꺼져 있습니다',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ),
            ),

            // ───── 불꽃 상태 ─────
            if (_flameF != null)
              FutureBuilder<int>(
                future: _flameF,
                builder: (ctx, fSnap) {
                  final txt = fSnap.connectionState != ConnectionState.done
                      ? '불꽃 상태 확인 중...'
                      : (fSnap.data == 0 ? '🔥 불꽃 감지!' : '정상');
                  return Padding(
                    padding: const EdgeInsets.all(8),
                    child: Text(txt, style: const TextStyle(fontSize: 18)),
                  );
                },
              ),

            // ───── 스트림 토글 버튼 ─────
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: FloatingActionButton.extended(
                onPressed: _toggleStream,
                icon: Icon(_streaming ? Icons.stop : Icons.play_arrow),
                label: Text(_streaming ? '중지' : '재생'),
              ),
            ),
          ],
        );
      },
    );
  }
}
// video_page.dart  ─ 변경·추가된 부분에 ★ 표시
import 'package:flutter/material.dart';
import 'api_service.dart';

class VideoPage extends StatefulWidget {
  const VideoPage({super.key});
  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  // ★ Map<String, String> → Map<String, dynamic> 로 변경
  late Future<Map<String, dynamic>> _devicesF; // ★

  String? _selected;

  // 스냅샷 URL과 이미지 로딩 상태를 관리
  String? _snapshotUrl;
  bool _isLoading = false;
  int _imageVersion = 0; // URL을 변경하여 새로고침을 강제하기 위한 변수

  Future<int>? _flameF;

  @override
  void initState() {
    super.initState();
    _devicesF = ApiService.fetchDevices(); // ★ 반환 타입과 일치
  }

  // 이미지 새로고침 함수
void _refreshSnapshot() {
  if (_selected == null) return;

  // 고유한 쿼리 파라미터(마이크로초)로 캐시 완전 회피
  final ts = DateTime.now().microsecondsSinceEpoch;

  // 이전 URL이 있었다면 캐시에서 제거(선택)
  if (_snapshotUrl != null) {
    final old = NetworkImage(_snapshotUrl!);
    old.evict(); // PaintingBinding.instance.imageCache.evict(...)
  }

  setState(() {
    _isLoading = true;
    _snapshotUrl = '${ApiService.snapshotUrl(_selected)}?t=$ts';
    // 여러 디바이스 지원하려면 fetchFlame(_selected!)로 변경 권장
    _flameF = ApiService.fetchFlame();
  });
}


  @override
  Widget build(BuildContext context) {
    // ★ FutureBuilder 제네릭도 Map<String, dynamic> 로 변경
    return FutureBuilder<Map<String, dynamic>>( // ★
      future: _devicesF,
      builder: (ctx, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snap.hasError || snap.data == null || snap.data!.isEmpty) {
          return Center(child: Text('장치가 없습니다: ${snap.error ?? ''}'));
        }

        // ★ /devices 응답에서 온라인 장치만 추출
        final Map<String, dynamic> all = snap.data!;
        final devices = all.entries
            .where((e) {
              final m = (e.value as Map<String, dynamic>? ?? {});
              final status = (m['status'] ?? '').toString().toLowerCase();
              return status == 'online';
            })
            .map((e) => e.key)
            .toList()
          ..sort();

        // ★ 선택된 장치가 오프라인으로 바뀐 경우 보정
        if (_selected == null || !devices.contains(_selected)) {
          _selected = devices.isNotEmpty ? devices.first : null;
          // 선택이 사라졌으니 스냅샷/불꽃 상태 초기화
          _snapshotUrl = null;
          _flameF = null;
        }

        // 온라인 장치가 하나도 없을 때
        if (_selected == null) {
          return Column(
            children: [
              const SizedBox(height: 16),
              const Text('현재 온라인인 장치가 없습니다.'),
              const SizedBox(height: 8),
              TextButton.icon(
                onPressed: () {
                  setState(() => _devicesF = ApiService.fetchDevices());
                },
                icon: const Icon(Icons.refresh),
                label: const Text('새로고침'),
              ),
            ],
          );
        }

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
                      _snapshotUrl = null; // 장치 변경 시 이미지 초기화
                      _flameF = null;
                    });
                  },
                ),
              ),
            ),

            // ───── 스냅샷 화면 / 대기화면 ─────
            Expanded(
              child: _snapshotUrl == null
                  ? const Center(
                      child: Text('아래 버튼을 눌러 이미지를 불러오세요.'),
                    )
                  : Image.network(
                      _snapshotUrl!,
                      key: ValueKey(_snapshotUrl), // URL이 바뀔 때마다 위젯을 새로 그리도록 키 설정
                      fit: BoxFit.contain,
                      loadingBuilder: (context, child, progress) {
                        if (progress == null) {
                          WidgetsBinding.instance.addPostFrameCallback((_) {
                            if (mounted && _isLoading) {
                              setState(() => _isLoading = false);
                            }
                          });
                          return child;
                        }
                        return const Center(child: CircularProgressIndicator());
                      },
                      errorBuilder: (context, error, stackTrace) {
                        WidgetsBinding.instance.addPostFrameCallback((_) {
                          if (mounted && _isLoading) {
                            setState(() => _isLoading = false);
                          }
                        });
                        return Center(child: Text('이미지 로딩 실패: $error'));
                      },
                    ),
            ),

            // ───── 불꽃 상태 ─────
            if (_flameF != null)
              FutureBuilder<int>(
                future: _flameF,
                builder: (ctx, fSnap) {
                  final txt = fSnap.connectionState != ConnectionState.done
                      ? '불꽃 상태 확인 중...'
                      : (fSnap.data == 0 ? '🔥 불꽃 감지!' : '정상'); // ★
                  return Padding(
                    padding: const EdgeInsets.all(8),
                    child: Text(txt, style: const TextStyle(fontSize: 18)),
                  );
                },
              ),

            // ───── 새로고침 버튼 ─────
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: FloatingActionButton.extended(
                onPressed: _isLoading ? null : _refreshSnapshot,
                icon: _isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.refresh),
                label: const Text('새로고침'),
              ),
            ),
          ],
        );
      },
    );
  }
}

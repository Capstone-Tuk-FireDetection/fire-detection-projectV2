import 'package:flutter/material.dart';
import 'api_service.dart';

class VideoPage extends StatefulWidget {
  const VideoPage({super.key});
  @override
  State<VideoPage> createState() => _VideoPageState();
}

class _VideoPageState extends State<VideoPage> {
  late Future<Map<String, dynamic>> _devicesF;

  String? _selected;

  // 스냅샷 URL과 이미지 로딩 상태를 관리
  String? _snapshotUrl;
  bool _isLoading = false;
  int _imageVersion = 0; // (원하면 삭제해도 됨: 현재 미사용)

  @override
  void initState() {
    super.initState();
    _devicesF = ApiService.fetchDevices();
  }

  // 이미지 새로고침 함수
  void _refreshSnapshot() {
    if (_selected == null) return;

    // 고유한 쿼리 파라미터(마이크로초)로 캐시 완전 회피
    final ts = DateTime.now().microsecondsSinceEpoch;

    // 이전 URL이 있었다면 캐시에서 제거(선택)
    if (_snapshotUrl != null) {
      final old = NetworkImage(_snapshotUrl!);
      old.evict();
    }

    setState(() {
      _isLoading = true;
      _snapshotUrl = '${ApiService.snapshotUrl(_selected)}?t=$ts';
    });
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Map<String, dynamic>>(
      future: _devicesF,
      builder: (ctx, snap) {
        if (snap.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snap.hasError || snap.data == null || snap.data!.isEmpty) {
          return Center(child: Text('장치가 없습니다: ${snap.error ?? ''}'));
        }

        // /devices 응답에서 온라인 장치만 추출
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

        // 선택된 장치가 오프라인으로 바뀐 경우 보정
        if (_selected == null || !devices.contains(_selected)) {
          _selected = devices.isNotEmpty ? devices.first : null;
          // 선택이 사라졌으니 이미지 초기화
          _snapshotUrl = null;
        }

        // 온라인 장치가 하나도 없을 때
        if (_selected == null) {
          return Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center, // 항상 중앙
            children: [
              const SizedBox(height: 16),
              const Text('현재 온라인인 장치가 없습니다.'),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.center,
                child: TextButton.icon(
                  onPressed: () {
                    setState(() => _devicesF = ApiService.fetchDevices());
                  },
                  icon: const Icon(Icons.refresh),
                  label: const Text('새로고침'),
                ),
              ),
            ],
          );
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.center, // ★ 가로 중앙 정렬 고정
          children: [
            // ───── 장치 선택 ─────
            Padding(
              padding: const EdgeInsets.all(8),
              child: Align(
                alignment: Alignment.center, // ★ 드롭다운 자체를 중앙
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    alignment: Alignment.center, // ★ 메뉴/라벨 정렬
                    value: _selected,
                    items: devices
                        .map((d) => DropdownMenuItem(
                              value: d,
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
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
                      });
                    },
                  ),
                ),
              ),
            ),

            // ───── 스냅샷 화면 / 대기화면 ─────
            Expanded(
              child: _snapshotUrl == null
                  ? const Center(child: Text('아래 버튼을 눌러 이미지를 불러오세요.'))
                  : Image.network(
                      _snapshotUrl!,
                      key: ValueKey(_snapshotUrl),
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

            // ───── 새로고침 버튼 ─────
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Align(
                alignment: Alignment.center, // ★ 버튼 중앙 고정
                child: ConstrainedBox(
                  constraints: const BoxConstraints(minWidth: 160), // ★ 폭 고정(선택)
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
              ),
            ),
          ],
        );
      },
    );
  }
}

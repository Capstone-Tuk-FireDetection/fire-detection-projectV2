import 'dart:async';
import 'package:flutter/material.dart';
import 'api_service.dart';
import 'constants.dart'; // backendBaseUrl

class RecordsPage extends StatefulWidget {
  const RecordsPage({super.key});
  @override
  State<RecordsPage> createState() => _RecordsPageState();
}

class _RecordsPageState extends State<RecordsPage> {
  late Future<List<Map<String, dynamic>>> _future;
  Timer? _poll;
  static const _pollInterval = Duration(seconds: 20);
  bool _deleting = false;

  @override
  void initState() {
    super.initState();
    _future = ApiService.fetchAlerts(limit: 50);
    _poll = Timer.periodic(_pollInterval, (_) => _refresh());
  }

  @override
  void dispose() {
    _poll?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    try {
      final data = await ApiService.fetchAlerts(limit: 50);
      if (!mounted) return;
      setState(() {
        _future = Future.value(data);
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _future = Future.error(e);
      });
    }
  }

  Future<void> _confirmAndDeleteAll() async {
    if (_deleting) return;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('전체 삭제'),
        content: const Text('최근 알람 기록을 모두 삭제하시겠어요?\n이 작업은 되돌릴 수 없습니다.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('취소')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('삭제')),
        ],
      ),
    );
    if (ok != true) return;

    setState(() => _deleting = true);
    try {
      final deleted = await ApiService.deleteAlerts(); // 전체 삭제
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('삭제됨: $deleted건')),
      );
      await _refresh();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('삭제 실패: $e')),
      );
    } finally {
      if (mounted) setState(() => _deleting = false);
    }
  }

  DateTime? _parseIso(String? s) {
    if (s == null || s.isEmpty) return null;
    try {
      return DateTime.parse(s).toLocal();
    } catch (_) {
      return null;
    }
  }

  String _timeAgo(DateTime when) {
    final diff = DateTime.now().difference(when);
    if (diff.inMinutes < 1) return '방금 전';
    if (diff.inMinutes < 60) return '${diff.inMinutes}분 전';
    final h = diff.inHours, m = diff.inMinutes % 60;
    if (m == 0) return '$h시간 전';
    return '$h시간 $m분 전';
  }

  void _openImageViewer(String fullUrl, String title) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => AlertImageViewerPage(imageUrl: fullUrl, title: title),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: _refresh,
      child: FutureBuilder<List<Map<String, dynamic>>>(
        future: _future,
        builder: (ctx, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Center(child: Text('불러오기 오류: ${snap.error}')),
                const SizedBox(height: 8),
                Center(
                  child: OutlinedButton.icon(
                    onPressed: _refresh,
                    icon: const Icon(Icons.refresh),
                    label: const Text('다시 시도'),
                  ),
                )
              ],
            );
          }

          final items = snap.data ?? const [];
          if (items.isEmpty) {
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Row(
                  children: [
                    Text('최근 알람 0건', style: Theme.of(context).textTheme.titleMedium),
                    const Spacer(),
                    IconButton(
                      onPressed: _deleting ? null : _confirmAndDeleteAll,
                      icon: _deleting
                          ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                          : const Icon(Icons.delete_forever),
                      tooltip: '전체 삭제',
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                const Center(child: Text('최근 알람이 없습니다.', style: TextStyle(color: Colors.grey))),
              ],
            );
          }

          return ListView.separated(
            padding: const EdgeInsets.all(8),
            itemCount: items.length + 2, // 헤더/구분선 위해 +2
            separatorBuilder: (_, __) => const Divider(height: 0),
            itemBuilder: (context, i) {
              if (i == 0) {
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                  child: Row(
                    children: [
                      Text('최근 알람 ${items.length}건', style: Theme.of(context).textTheme.titleMedium),
                      const Spacer(),
                      IconButton(
                        onPressed: _deleting ? null : _confirmAndDeleteAll,
                        icon: _deleting
                            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.delete_forever),
                        tooltip: '전체 삭제',
                      ),
                    ],
                  ),
                );
              }
              if (i == items.length + 1) {
                return const SizedBox(height: 12);
              }

              final it = items[i - 1];
              final when = _parseIso(it['created_at'] as String?) ?? DateTime.now();
              final device = (it['device'] ?? '(unknown)').toString();
              final title = (it['title'] ?? '알림').toString();
              final body = (it['body'] ?? '').toString();

              // 👇 image_url이 있으면 전체 URL 조합
              final imgPath = (it['image_url'] ?? '').toString();
              final imgUrl = imgPath.isNotEmpty ? '$backendBaseUrl$imgPath' : null;

              return ListTile(
                leading: const Icon(Icons.local_fire_department, color: Colors.red),
                title: Text('$title — $device'),
                subtitle: Text('$body\n${_timeAgo(when)}'),
                isThreeLine: true,
                trailing: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      TimeOfDay.fromDateTime(when).format(context),
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    if (imgUrl != null) ...[
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        icon: const Icon(Icons.photo),
                        label: const Text('사진'),
                        onPressed: () => _openImageViewer(imgUrl, '$title — $device'),
                      ),
                    ]
                  ],
                ),
                // 리스트 아이템 자체를 눌러도 열기
                onTap: imgUrl != null ? () => _openImageViewer(imgUrl, '$title — $device') : null,
              );
            },
          );
        },
      ),
    );
  }
}

class AlertImageViewerPage extends StatelessWidget {
  final String imageUrl;
  final String title;
  const AlertImageViewerPage({super.key, required this.imageUrl, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title, overflow: TextOverflow.ellipsis)),
      body: Center(
        child: InteractiveViewer(
          child: Image.network(
            imageUrl,
            fit: BoxFit.contain,
            loadingBuilder: (context, child, progress) {
              if (progress == null) return child;
              return const Center(child: CircularProgressIndicator());
            },
            errorBuilder: (context, err, st) {
              return const Center(child: Text('이미지를 불러오지 못했습니다.'));
            },
          ),
        ),
      ),
    );
  }
}

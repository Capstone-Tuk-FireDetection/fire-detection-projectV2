import 'dart:async';
import 'package:flutter/material.dart';
import 'api_service.dart';

class RecordsPage extends StatefulWidget {
  const RecordsPage({super.key});
  @override
  State<RecordsPage> createState() => _RecordsPageState();
}

class _RecordsPageState extends State<RecordsPage> {
  late Future<List<Map<String, dynamic>>> _future;
  Timer? _poll;
  static const _pollInterval = Duration(seconds: 20);

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
              children: const [
                SizedBox(height: 12),
                Center(child: Text('최근 알람이 없습니다.', style: TextStyle(color: Colors.grey))),
              ],
            );
          }

          return ListView.separated(
            padding: const EdgeInsets.all(8),
            itemCount: items.length,
            separatorBuilder: (_, __) => const Divider(height: 0),
            itemBuilder: (context, i) {
              final it = items[i];
              final when = _parseIso(it['created_at'] as String?) ?? DateTime.now();
              final device = (it['device'] ?? '(unknown)').toString();
              final title = (it['title'] ?? '알림').toString();
              final body = (it['body'] ?? '').toString();

              return ListTile(
                leading: const Icon(Icons.local_fire_department, color: Colors.red),
                title: Text('$title — $device'),
                subtitle: Text('$body\n${_timeAgo(when)}'),
                isThreeLine: true,
                trailing: Text(
                  TimeOfDay.fromDateTime(when).format(context),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              );
            },
          );
        },
      ),
    );
  }
}

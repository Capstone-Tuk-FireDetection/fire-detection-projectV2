import 'dart:async';
import 'package:flutter/material.dart';
import 'api_service.dart';

class DevicePage extends StatefulWidget {
  const DevicePage({super.key});

  @override
  State<DevicePage> createState() => _DevicePageState();
}

class _DevicePageState extends State<DevicePage> {
  late Future<Map<String, dynamic>> _future;

  // 최근 끊김 표시용(서버의 last_offline_at 우선, 없으면 최초 감지시각 기록)
  final Map<String, DateTime> _offlineAt = <String, DateTime>{};

  Timer? _pollTimer;
  Timer? _clockTimer;
  Timer? _aiStatusTimer; // Added
  static const _pollInterval = Duration(seconds: 20);
  static const _aiStatusPollInterval = Duration(seconds: 5); // Added

  String _aiStreamStatus = '확인 중...'; // Added
  bool _isAiStreamRunning = false; // Added
  bool _isAiActionPending = false; // Added

  @override
  void initState() {
    super.initState();
    _future = ApiService.fetchDevices();
    _startPolling();
    _startClock();
    _checkAiStreamStatus(); // Added
    _startAiStatusPolling(); // Added
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _clockTimer?.cancel();
    _aiStatusTimer?.cancel(); // Added
    super.dispose();
  }

  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(_pollInterval, (_) => _refreshDevices());
  }

  void _startClock() {
    _clockTimer?.cancel();
    _clockTimer = Timer.periodic(const Duration(minutes: 1), (_) {
      if (mounted && _offlineAt.isNotEmpty) setState(() {});
    });
  }

  DateTime? _parseIso(String? s) {
    if (s == null || s.isEmpty) return null;
    try {
      return DateTime.parse(s).toLocal();
    } catch (_) {
      return null;
    }
  }

  Future<void> _refreshDevices() async {
    try {
      final devices = await ApiService.fetchDevices();
      if (!mounted) return;

      // 서버 status/last_offline_at 기반으로 최근 끊김 갱신
      final now = DateTime.now();
      for (final entry in devices.entries) {
        final name = entry.key;
        final info = entry.value as Map<String, dynamic>? ?? {};
        final status = (info['status'] ?? '').toString().toLowerCase();
        final lastOffIso = info['last_offline_at'] as String?;
        final lastOff = _parseIso(lastOffIso);

        if (status == 'offline') {
          _offlineAt[name] = lastOff ?? _offlineAt[name] ?? now;
        } else {
          _offlineAt.remove(name); // 온라인이면 최근 끊김 목록에서 제거
        }
      }

      setState(() {
        _future = Future.value(devices);
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _future = Future.error(e);
        });
      }
    }
  }

        void _startAiStatusPolling() {
        _aiStatusTimer?.cancel();
        _aiStatusTimer = Timer.periodic(_aiStatusPollInterval, (_) => _checkAiStreamStatus());
      }

      Future<void> _checkAiStreamStatus() async {
        try {
          final status = await ApiService.getAiStreamStatus();
          if (!mounted) return;
          setState(() {
            _aiStreamStatus = status['status'] == 'running' ? '실행 중 (PID: ${status['pid']})' : '중지됨';
            _isAiStreamRunning = status['status'] == 'running';
          });
        } catch (e) {
          if (mounted) {
            setState(() {
              _aiStreamStatus = '상태 확인 오류: $e';
              _isAiStreamRunning = false;
            });
          }
        }
      }

      Future<void> _toggleAiStream() async {
        if (_isAiActionPending) return; // Prevent multiple clicks

        setState(() {
          _isAiActionPending = true;
          _aiStreamStatus = _isAiStreamRunning ? '중지 중...' : '시작 중...';
        });

        try {
          if (_isAiStreamRunning) {
            await ApiService.stopAiStream();
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('AI 스트림을 중지했습니다.')),
            );
          } else {
            await ApiService.startAiStream();
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('AI 스트림을 시작했습니다.')),
            );
          }
        } catch (e) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('AI 스트림 제어 오류: $e')),
          );
        } finally {
          if (mounted) {
            setState(() {
              _isAiActionPending = false;
            });
            _checkAiStreamStatus(); // Refresh status after action
          }
        }
      }

      String _timeAgo(DateTime when) {
        final diff = DateTime.now().difference(when);
        if (diff.inMinutes < 1) return '방금 전';
        if (diff.inMinutes < 60) return '${diff.inMinutes}분 전';
        final h = diff.inHours, m = diff.inMinutes % 60;
        if (m == 0) return '${h}시간 전';
        return '${h}시간 ${m}분 전';
      }

      String _timeAgo(DateTime when) {
        final diff = DateTime.now().difference(when);
        if (diff.inMinutes < 1) return '방금 전';
        if (diff.inMinutes < 60) return '${diff.inMinutes}분 전';
        final h = diff.inHours, m = diff.inMinutes % 60;
        if (m == 0) return '${h}시간 전';
        return '${h}시간 ${m}분 전';
      }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: _refreshDevices,
      child: FutureBuilder<Map<String, dynamic>>(
        future: _future,
        builder: (ctx, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Center(child: Text('오류: ${snap.error}')),
                const SizedBox(height: 12),
                Center(
                  child: OutlinedButton.icon(
                    onPressed: _refreshDevices,
                    icon: const Icon(Icons.refresh),
                    label: const Text('다시 시도'),
                  ),
                ),
              ],
            );
          }

          final devices = snap.data ?? const <String, dynamic>{};

          // online / offline 분리
          final online = <MapEntry<String, Map<String, dynamic>>>[];
          for (final e in devices.entries) {
            final info = (e.value as Map<String, dynamic>? ?? {});
            final status = (info['status'] ?? '').toString().toLowerCase();
            if (status == 'online') {
              online.add(MapEntry(e.key, info));
            }
          }

          final offlineList = _offlineAt.entries.toList()
            ..sort((a, b) => (b.value).compareTo(a.value)); // 최근 끊김 우선

          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // ===== AI 스트림 제어 =====
              Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: const [
                          Icon(Icons.psychology_alt),
                          SizedBox(width: 8),
                          Text('AI 스트림 제어', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text('상태: $_aiStreamStatus', style: const TextStyle(fontSize: 14)),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _isAiActionPending ? null : _toggleAiStream,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _isAiStreamRunning ? Colors.red : Colors.green,
                          foregroundColor: Colors.white,
                        ),
                        child: Text(_isAiStreamRunning ? 'AI 스트림 중지' : 'AI 스트림 시작'),
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // ===== 온라인 기기 =====
              Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.devices_other),
                          const SizedBox(width: 8),
                          const Text('연결된 기기', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                          const Spacer(),
                          IconButton(
                            tooltip: '새로고침',
                            onPressed: _refreshDevices,
                            icon: const Icon(Icons.refresh),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      if (online.isEmpty)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8),
                          child: Text('현재 연결된 기기가 없습니다.', style: TextStyle(color: Colors.grey)),
                        )
                      else
                        ...online.map((e) {
                          final ip = (e.value['ip'] ?? '').toString();
                          return ListTile(
                            dense: true,
                            leading: const Icon(Icons.videocam),
                            title: Text(e.key),
                            subtitle: Text(ip),
                            trailing: const Icon(Icons.circle, size: 10, color: Colors.green),
                          );
                        }),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // ===== 최근 끊긴 장치 =====
              Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(
                        children: const [
                          Icon(Icons.history),
                          SizedBox(width: 8),
                          Text('최근 끊긴 장치', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                        ],
                      ),
                      const SizedBox(height: 8),
                      if (offlineList.isEmpty)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8),
                          child: Text('최근 끊긴 장치가 없습니다.', style: TextStyle(color: Colors.grey)),
                        )
                      else
                        ...offlineList.map((e) {
                          final when = e.value;
                          return ListTile(
                            dense: true,
                            leading: const Icon(Icons.videocam_off),
                            title: Text(e.key),
                            subtitle: Text(_timeAgo(when)),
                            trailing: const Icon(Icons.circle, size: 10, color: Colors.grey),
                          );
                        }),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 8),
              Center(
                child: Text(
                  '자동 새로고침: ${_pollInterval.inSeconds}초 간격',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

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

  // 최근 끊김 표시용: 서버의 last_seen(마지막 접속 시각)을 우선 저장
  final Map<String, DateTime> _offlineAt = <String, DateTime>{};

  Timer? _pollTimer;
  Timer? _clockTimer;
  Timer? _aiStatusTimer;

  // 수동 스캔 진행 상태/타이머 (하단 배너 표시용)
  bool _isRescanning = false;
  Timer? _rescanTimer;
  static const _rescanPollInterval = Duration(seconds: 2);
  static const _rescanDuration = Duration(seconds: 20); // 스캔 기간 동안만 빠르게 폴링

  static const _pollInterval = Duration(seconds: 20);
  static const _aiStatusPollInterval = Duration(seconds: 5);

  String _aiStreamStatus = '확인 중...';
  bool _isAiStreamRunning = false;
  bool _isAiActionPending = false;

  @override
  void initState() {
    super.initState();
    _future = ApiService.fetchDevices();
    _startPolling();
    _startClock();
    _checkAiStreamStatus(); // 페이지 로드 시 상태 1회 확인
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _clockTimer?.cancel();
    _aiStatusTimer?.cancel();
    _rescanTimer?.cancel();
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

      // 서버 status / last_seen 기반으로 최근(오프라인) 목록 갱신
      final now = DateTime.now();
      for (final entry in devices.entries) {
        final name = entry.key;
        final info = (entry.value as Map<String, dynamic>? ?? {});
        final status = (info['status'] ?? '').toString().toLowerCase();

        // last_seen 사용: "마지막까지 있었던 시각"
        final lastSeenIso = info['last_seen'] as String?;
        final lastSeen = _parseIso(lastSeenIso);

        if (status == 'offline') {
          _offlineAt[name] = lastSeen ?? _offlineAt[name] ?? now;
        } else {
          _offlineAt.remove(name); // 온라인이면 최근 목록에서 제거
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

  // 수동 스캔 트리거 (+ 버튼)
  Future<void> _rescanForNewDevices() async {
    if (_isRescanning) return;
    setState(() => _isRescanning = true);

    try {
      await ApiService.rescanDevices(); // 서버 측 network_scanner 비동기 실행
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('새 기기 검색 시작 실패: $e')),
      );
      setState(() => _isRescanning = false);
      return;
    }

    // 스캔 기간 동안 빠르게 폴링
    _rescanTimer?.cancel();
    final startedAt = DateTime.now();
    _rescanTimer = Timer.periodic(_rescanPollInterval, (t) async {
      await _refreshDevices();
      if (DateTime.now().difference(startedAt) >= _rescanDuration) {
        t.cancel();
        if (mounted) setState(() => _isRescanning = false);
      }
    });
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
    if (_isAiActionPending) return;

    setState(() {
      _isAiActionPending = true;
      _aiStreamStatus = _isAiStreamRunning ? '중지 중...' : '시작 중...';
    });

    try {
      if (_isAiStreamRunning) {
        await ApiService.stopAiStream();
        _aiStatusTimer?.cancel();
        _startPolling();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('AI 스트림을 중지했습니다.')),
        );
      } else {
        await ApiService.startAiStream();
        _startAiStatusPolling();
        _pollTimer?.cancel();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('AI 스트림을 시작했습니다.')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('AI 스트림 제어 오류: $e')),
      );
    } finally {
      if (mounted) {
        setState(() {
          _isAiActionPending = false;
        });
        _checkAiStreamStatus();
      }
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

  // 하단 고정 배너 위젯
  Widget _buildBottomRescanBanner(BuildContext context) {
    return SafeArea(
      minimum: const EdgeInsets.only(left: 16, right: 16, bottom: 16),
      child: Material(
        elevation: 6,
        borderRadius: BorderRadius.circular(12),
        color: Theme.of(context).colorScheme.surface,
        child: const ListTile(
          leading: SizedBox(
            width: 20,
            height: 20,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          title: Text('새로운 기기 검색중…'),
          dense: true,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // 메인 콘텐츠 (ListView 포함)
    final content = RefreshIndicator(
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

              // ===== 연결된 기기 =====
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
                          // 새 기기 스캔(등록) 버튼
                          IconButton(
                            tooltip: '새 기기 검색',
                            onPressed: _isRescanning ? null : _rescanForNewDevices,
                            icon: const Icon(Icons.add),
                          ),
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

              // ===== 최근 장치 (오프라인) =====
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
                          Text('최근 장치', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                        ],
                      ),
                      const SizedBox(height: 8),
                      if (offlineList.isEmpty)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8),
                          child: Text('최근 장치가 없습니다.', style: TextStyle(color: Colors.grey)),
                        )
                      else
                        ...offlineList.map((e) {
                          final name = e.key;
                          final whenFallback = e.value;
                          final info = (devices[name] as Map<String, dynamic>? ?? {});
                          final ip = (info['ip'] ?? '').toString();
                          final lastSeen = _parseIso(info['last_seen'] as String?) ?? whenFallback;

                          return ListTile(
                            dense: true,
                            leading: const Icon(Icons.videocam_off),
                            title: Text(name),
                            subtitle: Text(ip.isNotEmpty ? '$ip · ${_timeAgo(lastSeen)}' : _timeAgo(lastSeen)),
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
              const SizedBox(height: 80), // 하단 배너와 겹치지 않도록 여유 공간
            ],
          );
        },
      ),
    );

    // 하단 고정 배너를 오버레이로 노출
    return Stack(
      children: [
        content,
        if (_isRescanning)
          Align(
            alignment: Alignment.bottomCenter,
            child: _buildBottomRescanBanner(context),
          ),
      ],
    );
  }
}

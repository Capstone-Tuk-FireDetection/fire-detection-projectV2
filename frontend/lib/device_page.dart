import 'package:flutter/material.dart';
import 'api_service.dart';

class DevicePage extends StatefulWidget {
  const DevicePage({super.key});

  @override
  State<DevicePage> createState() => _DevicePageState();
}

class _DevicePageState extends State<DevicePage> {
  late Future<Map<String, String>> _future;

  @override
  void initState() {
    super.initState();
    _future = ApiService.fetchDevices();
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: () async => setState(() => _future = ApiService.fetchDevices()),
      child: FutureBuilder<Map<String, String>>(
        future: _future,
        builder: (ctx, snap) {
          if (snap.connectionState != ConnectionState.done) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snap.hasError) {
            return Center(child: Text('오류: ${snap.error}'));
          }
          final devices = snap.data!;
          if (devices.isEmpty) {
            return const Center(child: Text('연결된 기기가 없습니다'));
          }
          return ListView(
            children: devices.entries
                .map((e) => ListTile(
                      leading: const Icon(Icons.videocam),
                      title: Text(e.key),
                      subtitle: Text(e.value),
                    ))
                .toList(),
          );
        },
      ),
    );
  }
}

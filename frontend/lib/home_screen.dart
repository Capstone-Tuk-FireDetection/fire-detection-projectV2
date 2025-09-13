import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'device_page.dart';
import 'video_page.dart';
import 'records_page.dart'; // ★ 추가
import 'user_page.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _idx = 0;
  // ★ 페이지 배열에 기록 탭 추가 (영상과 사용자 사이)
  final _pages = const [DevicePage(), VideoPage(), RecordsPage(), UserPage()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('화재 감지 시스템'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () => FirebaseAuth.instance.signOut(),
            tooltip: '로그아웃',
          ),
        ],
      ),
      body: _pages[_idx],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _idx,
        destinations: const [
          NavigationDestination(icon: Icon(Icons.sensors), label: '장치'),
          NavigationDestination(icon: Icon(Icons.videocam), label: '영상'),
          NavigationDestination(icon: Icon(Icons.history), label: '기록'), // ★ 추가
          NavigationDestination(icon: Icon(Icons.person), label: '사용자'),
        ],
        onDestinationSelected: (i) => setState(() => _idx = i),
      ),
    );
  }
}

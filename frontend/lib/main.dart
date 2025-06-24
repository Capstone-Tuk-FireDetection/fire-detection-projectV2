import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:http/http.dart' as http;

import 'firebase_options.dart';
import 'login_screen.dart';
import 'home_screen.dart';
import 'constants.dart'; // backendBaseUrl, vapidKey 정의

// ✅ 글로벌 키로 ScaffoldMessenger 상태 추적
final GlobalKey<ScaffoldMessengerState> scaffoldMessengerKey =
    GlobalKey<ScaffoldMessengerState>();

Future<void> registerFcmTokenIfLoggedIn() async {
  try {
    final notifSettings = await FirebaseMessaging.instance.requestPermission();
    if (notifSettings.authorizationStatus != AuthorizationStatus.authorized) {
      debugPrint('❌ 알림 권한 거부됨');
      return;
    }

    final fcmToken = await FirebaseMessaging.instance.getToken(vapidKey: vapidKey);
    if (fcmToken == null) {
      debugPrint('❌ FCM 토큰 획득 실패');
      return;
    }

    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      debugPrint('⚠️ 로그인 상태 아님. 토큰 서버 전송 보류');
      return;
    }

    final idToken = await user.getIdToken();
    final res = await http.post(
      Uri.parse('$backendBaseUrl/register_token'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $idToken',
      },
      body: jsonEncode({'token': fcmToken}),
    );

    debugPrint('📨 FCM 등록 응답: ${res.statusCode} ${res.body}');
  } catch (e) {
    debugPrint('❌ FCM 등록 중 오류: $e');
  }
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  runApp(const FireCamApp());
}

class FireCamApp extends StatefulWidget {
  const FireCamApp({super.key});
  @override
  State<FireCamApp> createState() => _FireCamAppState();
}

class _FireCamAppState extends State<FireCamApp> {
  @override
  void initState() {
    super.initState();

    // ✅ 포그라운드 알림 수신 시 UI 알림 표시
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      print('📩 FCM 수신됨 (포그라운드)');
      print('▶️ message: ${message.toMap()}');

      final notification = message.notification;
      if (notification == null) {
        print('⚠️ notification 없음');
        return;
      }

      // ✅ GlobalKey로 SnackBar 안전하게 표시
      final snackBar = SnackBar(
        content: Text(
          '${notification.title ?? "알림"}\n${notification.body ?? ""}',
        ),
        duration: const Duration(seconds: 5),
        behavior: SnackBarBehavior.floating,
        backgroundColor: Colors.redAccent,
      );

      scaffoldMessengerKey.currentState?.showSnackBar(snackBar);
    });

    // ✅ 토큰 갱신 시 재등록
    FirebaseMessaging.instance.onTokenRefresh.listen((newToken) {
      registerFcmTokenIfLoggedIn();
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart FireCam',
      scaffoldMessengerKey: scaffoldMessengerKey, // ✅ 여기 중요!
      theme: ThemeData(
        colorSchemeSeed: Colors.deepOrange,
        useMaterial3: true,
      ),
      home: StreamBuilder<User?>(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Scaffold(
              body: Center(child: CircularProgressIndicator()),
            );
          }

          if (snap.data == null) {
            return const LoginScreen();
          }

          // ✅ 로그인 완료 시 FCM 토큰 등록
          registerFcmTokenIfLoggedIn();
          return const HomeScreen();
        },
      ),
    );
  }
}

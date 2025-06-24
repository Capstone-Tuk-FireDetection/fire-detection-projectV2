import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserPage extends StatelessWidget {
  const UserPage({super.key});

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    return Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        const Icon(Icons.person, size: 64),
        Text(user?.email ?? 'Unknown', style: const TextStyle(fontSize: 20)),
        const SizedBox(height: 16),
        /*  {{탈퇴}} 구현 자리 */
      ]),
    );
  }
}

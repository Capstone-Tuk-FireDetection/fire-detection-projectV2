import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _formKey = GlobalKey<FormState>();

  String _email = '';
  String _pw = '';
  String _pwConfirm = '';
  bool _loading = false;
  bool _obscure1 = true;
  bool _obscure2 = true;

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    _formKey.currentState!.save();

    if (_pw != _pwConfirm) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('비밀번호가 일치하지 않습니다.')),
      );
      return;
    }

    setState(() => _loading = true);

    try {
      // Firebase에 사용자 생성
      await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: _email,
        password: _pw,
      );

      if (!mounted) return;

      // 성공 시 알림 후 로그인 화면으로 복귀
      await showDialog<void>(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('회원가입 완료'),
          content: Text('계정이 성공적으로 생성되었습니다: $_email\n로그인 페이지로 이동합니다.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('확인'),
            ),
          ],
        ),
      );
      
      if (mounted) Navigator.of(context).pop(true);

    } on FirebaseAuthException catch (e) {
      // 오류 처리
      String message = '오류가 발생했습니다.';
      if (e.code == 'weak-password') {
        message = '비밀번호가 너무 약합니다.';
      } else if (e.code == 'email-already-in-use') {
        message = '이미 사용 중인 이메일입니다.';
      } else if (e.code == 'invalid-email') {
        message = '유효하지 않은 이메일 형식입니다.';
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('알 수 없는 오류: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('회원가입')),
      body: Center(
        child: Card(
          margin: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('회원가입', style: TextStyle(fontSize: 22)),
                  const SizedBox(height: 12),
                  TextFormField(
                    decoration: const InputDecoration(labelText: 'Email'),
                    keyboardType: TextInputType.emailAddress,
                    validator: (v) =>
                        (v == null || v.trim().isEmpty) ? '필수 입력' : null,
                    onSaved: (v) => _email = v!.trim(),
                  ),
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Password (6자 이상)',
                      suffixIcon: IconButton(
                        icon: Icon(_obscure1 ? Icons.visibility : Icons.visibility_off),
                        onPressed: () => setState(() => _obscure1 = !_obscure1),
                      ),
                    ),
                    obscureText: _obscure1,
                    validator: (v) =>
                        (v == null || v.length < 6) ? '6자 이상 입력' : null,
                    onSaved: (v) => _pw = v!.trim(),
                  ),
                  TextFormField(
                    decoration: InputDecoration(
                      labelText: 'Confirm Password',
                      suffixIcon: IconButton(
                        icon: Icon(_obscure2 ? Icons.visibility : Icons.visibility_off),
                        onPressed: () => setState(() => _obscure2 = !_obscure2),
                      ),
                    ),
                    obscureText: _obscure2,
                    validator: (v) =>
                        (v == null || v.isEmpty) ? '필수 입력' : null,
                    onSaved: (v) => _pwConfirm = v!.trim(),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      TextButton(
                        onPressed: () => Navigator.of(context).maybePop(),
                        child: const Text('취소'),
                      ),
                      _loading
                          ? const CircularProgressIndicator()
                          : ElevatedButton(
                              onPressed: _submit,
                              child: const Text('회원가입'),
                            ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
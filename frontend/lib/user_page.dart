import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserPage extends StatefulWidget {
  const UserPage({super.key});

  @override
  State<UserPage> createState() => _UserPageState();
}

class _UserPageState extends State<UserPage> {
  final _formKey = GlobalKey<FormState>();
  final _currentPwCtrl = TextEditingController();
  final _newPwCtrl = TextEditingController();
  final _confirmPwCtrl = TextEditingController();

  bool _obscureCurrent = true;
  bool _obscureNew = true;
  bool _obscureConfirm = true;
  bool _loading = false;

  @override
  void dispose() {
    _currentPwCtrl.dispose();
    _newPwCtrl.dispose();
    _confirmPwCtrl.dispose();
    super.dispose();
  }

  User? get _user => FirebaseAuth.instance.currentUser;

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  String _mapAuthError(FirebaseAuthException e) {
    switch (e.code) {
      case 'wrong-password':
        return '현재 비밀번호가 올바르지 않습니다.';
      case 'user-mismatch':
        return '사용자 정보가 일치하지 않습니다.';
      case 'user-not-found':
        return '사용자를 찾을 수 없습니다.';
      case 'weak-password':
        return '새 비밀번호가 안전하지 않습니다. 더 복잡하게 설정하세요.';
      case 'requires-recent-login':
        return '보안을 위해 최근 로그인(재인증)이 필요합니다. 현재 비밀번호로 다시 인증해주세요.';
      default:
        return '작업을 완료할 수 없습니다: ${e.message ?? e.code}';
    }
  }

  Future<void> _reauthenticateWithPassword(String email, String currentPassword) async {
    final cred = EmailAuthProvider.credential(email: email, password: currentPassword);
    await _user!.reauthenticateWithCredential(cred);
  }

  Future<void> _changePassword() async {
    if (_user == null) {
      _showSnack('로그인이 필요합니다.');
      return;
    }
    if (!_formKey.currentState!.validate()) return;

    final email = _user!.email;
    final currentPw = _currentPwCtrl.text.trim();
    final newPw = _newPwCtrl.text.trim();

    setState(() => _loading = true);
    try {
      await _reauthenticateWithPassword(email!, currentPw);
      await _user!.updatePassword(newPw);
      _showSnack('비밀번호가 변경되었습니다. 다음 로그인부터 새 비밀번호를 사용하세요.');

      _currentPwCtrl.clear();
      _newPwCtrl.clear();
      _confirmPwCtrl.clear();
    } on FirebaseAuthException catch (e) {
      _showSnack(_mapAuthError(e));
    } catch (e) {
      _showSnack('알 수 없는 오류가 발생했습니다: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _signOut() async {
    setState(() => _loading = true);
    try {
      await FirebaseAuth.instance.signOut();
    } catch (e) {
      _showSnack('로그아웃에 실패했습니다: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _deleteAccount() async {
    final user = _user;
    if (user == null) {
      _showSnack('로그인이 필요합니다.');
      return;
    }

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('계정을 삭제할까요?'),
        content: const Text('계정과 관련 데이터가 모두 삭제될 수 있어요. 이 작업은 되돌릴 수 없습니다.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('취소')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('삭제')),
        ],
      ),
    );

    if (confirmed != true) return;

    final email = user.email ?? '';
    final pwCtrl = TextEditingController();
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('본인 확인'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(email, style: const TextStyle(fontWeight: FontWeight.w600)),
            const SizedBox(height: 12),
            TextField(
              controller: pwCtrl,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: '현재 비밀번호',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('취소')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('확인')),
        ],
      ),
    );
    if (ok != true) return;

    setState(() => _loading = true);
    try {
      await _reauthenticateWithPassword(email, pwCtrl.text.trim());
      await user.delete();
      _showSnack('계정이 삭제되었습니다.');
    } on FirebaseAuthException catch (e) {
      _showSnack(_mapAuthError(e));
    } catch (e) {
      _showSnack('계정 삭제 중 오류가 발생했습니다: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = _user;

    return Stack(
      children: [
        SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 560),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 계정 정보 (이메일 인증 표시 제거)
                  Card(
                    clipBehavior: Clip.antiAlias,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Row(
                        children: [
                          const CircleAvatar(
                            radius: 32,
                            child: Icon(Icons.person, size: 36),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  user?.email ?? 'Unknown',
                                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  'UID: ${user?.uid ?? '-'}',
                                  style: TextStyle(color: Theme.of(context).textTheme.bodySmall?.color),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // 보안: 비밀번호 변경
                  Card(
                    clipBehavior: Clip.antiAlias,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Form(
                        key: _formKey,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            const Text('비밀번호 변경', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                            const SizedBox(height: 12),

                            TextFormField(
                              controller: _currentPwCtrl,
                              obscureText: _obscureCurrent,
                              decoration: InputDecoration(
                                labelText: '현재 비밀번호',
                                border: const OutlineInputBorder(),
                                suffixIcon: IconButton(
                                  onPressed: () => setState(() => _obscureCurrent = !_obscureCurrent),
                                  icon: Icon(_obscureCurrent ? Icons.visibility : Icons.visibility_off),
                                  tooltip: _obscureCurrent ? '표시' : '숨기기',
                                ),
                              ),
                              validator: (v) {
                                if (v == null || v.trim().isEmpty) return '현재 비밀번호를 입력하세요.';
                                return null;
                              },
                            ),
                            const SizedBox(height: 12),

                            TextFormField(
                              controller: _newPwCtrl,
                              obscureText: _obscureNew,
                              decoration: InputDecoration(
                                labelText: '새 비밀번호 (6자 이상)',
                                border: const OutlineInputBorder(),
                                suffixIcon: IconButton(
                                  onPressed: () => setState(() => _obscureNew = !_obscureNew),
                                  icon: Icon(_obscureNew ? Icons.visibility : Icons.visibility_off),
                                  tooltip: _obscureNew ? '표시' : '숨기기',
                                ),
                              ),
                              validator: (v) {
                                final text = v?.trim() ?? '';
                                if (text.length < 6) return '6자 이상으로 설정하세요.';
                                if (text == _currentPwCtrl.text.trim()) return '현재 비밀번호와 달라야 합니다.';
                                return null;
                              },
                            ),
                            const SizedBox(height: 12),

                            TextFormField(
                              controller: _confirmPwCtrl,
                              obscureText: _obscureConfirm,
                              decoration: InputDecoration(
                                labelText: '새 비밀번호 확인',
                                border: const OutlineInputBorder(),
                                suffixIcon: IconButton(
                                  onPressed: () => setState(() => _obscureConfirm = !_obscureConfirm),
                                  icon: Icon(_obscureConfirm ? Icons.visibility : Icons.visibility_off),
                                  tooltip: _obscureConfirm ? '표시' : '숨기기',
                                ),
                              ),
                              validator: (v) {
                                if (v?.trim() != _newPwCtrl.text.trim()) return '비밀번호가 일치하지 않습니다.';
                                return null;
                              },
                            ),
                            const SizedBox(height: 12),

                            FilledButton.icon(
                              onPressed: _loading ? null : _changePassword,
                              icon: const Icon(Icons.lock_reset),
                              label: const Text('비밀번호 변경'),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // 계정 작업
                  Card(
                    clipBehavior: Clip.antiAlias,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const Text('계정', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700)),
                          const SizedBox(height: 12),
                          Row(
                            children: [
                              Expanded(
                                child: OutlinedButton.icon(
                                  onPressed: _loading ? null : _signOut,
                                  icon: const Icon(Icons.logout),
                                  label: const Text('로그아웃'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: FilledButton.tonalIcon(
                                  style: FilledButton.styleFrom(
                                    foregroundColor: Colors.red,
                                  ),
                                  onPressed: _loading ? null : _deleteAccount,
                                  icon: const Icon(Icons.delete_forever),
                                  label: const Text('탈퇴'),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        if (_loading)
          Positioned.fill(
            child: AbsorbPointer(
              child: Container(
                color: Colors.black.withOpacity(0.08),
                child: const Center(child: CircularProgressIndicator()),
              ),
            ),
          ),
      ],
    );
  }
}

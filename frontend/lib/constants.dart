/// backendBaseUrl
/// ─────────────────────────────────────────────────
/// ▸ Android ‘emulator’ 로 앱을 돌릴 때
///   - 10.0.2.2 는 호스트 PC 의 127.0.0.1 로 NAT-포워딩
/// ▸ 실제 스마트폰(동일 Wi-Fi)이나 iOS 시뮬레이터에서 테스트할 때
///   - PC 가 받은 사설 IP(여기선 192.168.218.34)를 사용
///
/// 두 환경을 번갈아 쓰면 하나만 살리고 나머지는 주석 처리하세요.

/// ――― 에뮬레이터 전용 ―――
/// const String backendBaseUrl = 'http://10.0.2.2:8080';

/// ――― 물리 단말/다른 PC/동일 LAN ―――
const String backendBaseUrl = 'http://localhost:8080';
const String vapidKey = 'BKKkmes42U0zpwWAW9UkaNLiyz3vNxbUw7-iXML7n3asNksAvAVuRgm4D_0oxaFHLYeEmrShBDygcK-8ZvUDxeE';

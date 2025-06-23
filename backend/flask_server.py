# flask_server_combined.py
from flask import Flask, jsonify, Response, stream_with_context, request
import requests
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
from firebase_admin import messaging
<<<<<<< Updated upstream
# flask_server.py 맨 위쪽
from flask import Flask, jsonify, request, Response
from flask_cors import CORS               # ← 추가
=======
import subprocess
>>>>>>> Stashed changes

app = Flask(__name__)
# 모든 경로에 모든 Origin 허용 (개발용)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# ✅ Firebase Admin 초기화
<<<<<<< Updated upstream
cred = credentials.Certificate("firebase-adminsdk.json")
=======
cred = credentials.Certificate("./firebase-adminsdk.json")
>>>>>>> Stashed changes
firebase_admin.initialize_app(cred)

# ✅ 메모리 기반 저장소
registered_devices = {}      # 공개용 디바이스 (mDNS 기반)
index=0
user_devices = {}            # 사용자별 디바이스 (인증 필요)
user_fcm_tokens = {}         # 사용자별 FCM 토큰 저장

# ✅ FCM 알림 전송 함수
def send_fcm_notification(token, title, body):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body
        ),
        token=token
    )
    try:
        response = messaging.send(message)
        print(f"✅ FCM 메시지 전송됨: {response}")
    except Exception as e:
        print(f"❌ FCM 전송 실패: {e}")


<<<<<<< Updated upstream
# ✅ mDNS lookup 기능 통합


def resolve_mdns(name, timeout=3):
    # 1) OS가 이미 .local 이름을 해석할 수 있으면 바로 사용
    try:
        return socket.gethostbyname(f"{name}.local")
    except socket.gaierror:
        pass                     # 실패하면 Zeroconf로 넘어감

    # 2) Zeroconf로 정확히 ‘name’만 필터링
    class MDNSListener(ServiceListener):
        def __init__(self, target):
            self.address = None
            self.target = f"{target}._http._tcp.local."

        def add_service(self, zeroconf, type, svc_name):
            if svc_name != self.target:
                return            # 다른 장치 패스
            info = zeroconf.get_service_info(type, svc_name, timeout=1000)
            if info and info.addresses:
                self.address = socket.inet_ntoa(info.addresses[0])

    zeroconf = Zeroconf()
    listener = MDNSListener(name)
    ServiceBrowser(zeroconf, "_http._tcp.local.", listener)

    for _ in range(timeout * 10):
        if listener.address:
            zeroconf.close()
            return listener.address
        time.sleep(0.1)

    zeroconf.close()
    return None


=======
>>>>>>> Stashed changes
# ✅ Firebase 토큰 검증 데코레이터
def firebase_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing"}), 401
        id_token = auth_header.split(" ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.uid = decoded_token['uid']
        except Exception as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401
        return f(*args, **kwargs)
    return decorated

# ✅ FCM 토큰 등록 API
@app.route("/register_token", methods=["POST"])
@firebase_required
def register_fcm_token():
    data = request.json
    token = data.get("token")
    if not token:
        return jsonify({"error": "FCM token required"}), 400
    user_fcm_tokens[request.uid] = token
    return jsonify({"status": "token registered"})

# ✅ 공개 디바이스 등록 (ESP32에서 호출)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    ip = data.get("ip")
    name = data.get("device_name")
    if ip:
        index = index + 1
        registered_devices[name] = ip
         # 등록된 디바이스에 대해 AI 분석 프로세스 실행
        try:
            subprocess.Popen(
                ["python", "stream_flame_detection.py", "--ip", ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"🚀 AI 프로세스 시작: {ip}")
        except Exception as e:
            print(f"❌ AI 프로세스 실행 실패: {e}")
        return jsonify({"status": "ok", "device_id": index, "device_name":name, "device_ip":ip})
    return jsonify({"error": "Invalid payload"}), 400

# ✅ 공개 디바이스 목록 조회
@app.route("/devices")
def list_devices():
    return jsonify(registered_devices)

# ✅ 사용자 디바이스 등록 (Firebase 로그인 필요)
@app.route("/user/devices", methods=["POST"])
@firebase_required
def register_user_device():
    data = request.json
    device_name = data.get("device_name")
    ip = data.get("ip")
    if not device_name or not ip:
        return jsonify({"error": "device_name and ip required"}), 400

    user_devices.setdefault(request.uid, {})[device_name] = ip
    return jsonify({"status": "registered", "device_name": device_name})

# ✅ 사용자 디바이스 목록 조회
@app.route("/user/devices", methods=["GET"])
@firebase_required
def get_user_devices():
    return jsonify(user_devices.get(request.uid, {}))

# ✅ 사용자 디바이스 삭제
@app.route("/user/devices/<device_name>", methods=["DELETE"])
@firebase_required
def delete_user_device(device_name):
    devices = user_devices.get(request.uid, {})
    if device_name in devices:
        del devices[device_name]
        return jsonify({"status": "deleted", "device_name": device_name})
    return jsonify({"error": "Device not found"}), 404

# ✅ mDNS 기반 디바이스 스트리밍 프록시
@app.route("/flame")
def get_flame():
    resolved_ip = resolve_mdns("espcam2") or registered_devices.get("espcam2")
    if not resolved_ip:
        return jsonify({"flame": -1, "error": "Device not found"}), 404
    try:
        resp = requests.get(f"http://{resolved_ip}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503

@app.route("/stream")
def stream():
    resolved_ip = resolve_mdns("espcam2") or registered_devices.get("espcam2")
    if not resolved_ip:
        return Response("Device not found", status=404)
    try:
        r = requests.get(f"http://{resolved_ip}/stream", stream=True, timeout=5)
        return Response(
            stream_with_context(r.iter_content(chunk_size=1024)),
            content_type=r.headers.get("Content-Type", "multipart/x-mixed-replace")
        )
    except requests.RequestException as e:
        return Response(f"Stream error: {str(e)}", status=503)

# ✅ 디바이스 이름별 스트림 프록시
@app.route("/stream/<device>")
def stream_device(device):
    resolved_ip = registered_devices.get(device) or resolve_mdns(device)
    if not resolved_ip:
        return Response("Device not found", status=404)
    try:
        r = requests.get(f"http://{resolved_ip}/stream", stream=True, timeout=5)
        return Response(
            stream_with_context(r.iter_content(chunk_size=1024)),
            content_type=r.headers.get("Content-Type", "multipart/x-mixed-replace")
        )
    except requests.RequestException as e:
        return Response(f"Stream error: {str(e)}", status=503)

# ✅ AI+센서 감지 시 알림 전송용 API
@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json()
    if data.get("flame") == 1:
        device = data.get("device", "(unknown)")
        print(f"🔥 불꽃 감지됨! [디바이스: {device}]")
        for token in user_fcm_tokens.items():
            send_fcm_notification(token, "불꽃 감지", f"🔥 {device} 장치에서 불꽃이 감지되었습니다.")
    return jsonify({"received": True})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

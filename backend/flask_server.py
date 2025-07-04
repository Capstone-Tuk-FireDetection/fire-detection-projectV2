from flask import Flask, jsonify, request, Response, stream_with_context
import requests
import firebase_admin
from firebase_admin import credentials, auth, messaging
from functools import wraps
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Firebase 초기화
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# ✅ 메모리 저장소
registered_devices = {}
user_devices = {}
fcm_tokens = []
device_index = 0  # 전역 인덱스

# ✅ FCM 알림 함수
def send_fcm_notification_to_all(title, body):
    for token in fcm_tokens:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token
        )
        try:
            response = messaging.send(message)
            print(f"✅ 전송 완료: {token[:16]}... → {response}")
        except Exception as e:
            print(f"❌ 전송 실패: {e}")

# ✅ Firebase 인증 데코레이터
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

# ✅ FCM 토큰 등록
@app.route("/register_token", methods=["POST"])
def register_fcm_token():
    data = request.json
    token = data.get("token")
    if not token:
        return jsonify({"error": "FCM token required"}), 400

    fcm_tokens.append(token)
    print(f"🔔 공개 등록된 FCM 토큰: {token[:16]}...")  # 로그 일부만 출력
    return jsonify({"status": "token registered (public)"})

# ✅ 디바이스 등록
@app.route("/register", methods=["POST"])
def register():
    global device_index
    data = request.json
    ip = data.get("ip")
    name = data.get("device_name")
    if not ip or not name:
        return jsonify({"error": "Invalid payload"}), 400

    device_index += 1
    registered_devices[name] = ip
    return jsonify({"status": "ok", "device_id": device_index, "device_name": name, "device_ip": ip})

"""
    # AI 분석 프로세스 실행
    try:
        subprocess.Popen(
            ["python", "stream_flame_detection.py", "--ip", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"🚀 AI 프로세스 시작: {ip}")
    except Exception as e:
        print(f"❌ AI 프로세스 실행 실패: {e}")

"""

# ✅ 디바이스 목록 조회
@app.route("/devices")
def list_devices():
    return jsonify(registered_devices)

# ✅ 사용자 디바이스 등록
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

# ✅ 사용자 디바이스 목록
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

# ✅ flame 상태 조회 (기본 디바이스 "espcam2")
@app.route("/flame")
def get_flame():
    ip = registered_devices.get("espcam2")
    if not ip:
        return jsonify({"flame": -1, "error": "Device not found"}), 404
    try:
        resp = requests.get(f"http://{ip}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503

@app.route("/flame/<device>")
def get_flame_by_name(device):
    ip = registered_devices.get(device)
    if not ip:
        return jsonify({"flame": -1, "error": "Device not found"}), 404
    try:
        resp = requests.get(f"http://{ip}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503


# ✅ 스트리밍 (기본 디바이스)
@app.route("/stream")
def stream():
    ip = registered_devices.get("espcam2")
    if not ip:
        return Response("Device not found", status=404)
    try:
        r = requests.get(f"http://{ip}/stream", stream=True, timeout=5)
        return Response(
            stream_with_context(r.iter_content(chunk_size=1024)),
            content_type=r.headers.get("Content-Type", "multipart/x-mixed-replace")
        )
    except requests.RequestException as e:
        return Response(f"Stream error: {str(e)}", status=503)

# 디바이스별 스트리밍 요청 상태 캐싱 (스레드 안전하게)
active_stream_clients = {}
lock = threading.Lock()

@app.route("/stream/<device>")
def stream_device(device):
    #if request.remote_addr.startswith("192.168.0.51"):  # Flutter 앱 IP라면
     #   return Response("Flutter 접근 차단", status=403)
    
    ip = registered_devices.get(device)
    if not ip:
        return Response("Device not found", status=404)

    with lock:
        if active_stream_clients.get(device):
            return Response("❌ 해당 디바이스는 이미 스트리밍 중입니다.", status=429)
        active_stream_clients[device] = True

    def generate():
        try:
            r = requests.get(f"http://{ip}/stream", stream=True, timeout=5)
            for chunk in r.iter_content(chunk_size=1024):
                yield chunk
        except Exception as e:
            print(f"❌ 스트리밍 오류({device}):", e)
        finally:
            with lock:
                active_stream_clients[device] = False
            print(f"🔁 스트리밍 종료: {device}")

    return Response(stream_with_context(generate()),
                    content_type="multipart/x-mixed-replace; boundary=frame")
    


# ✅ 알림 수신 API
@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json()
    if data.get("flame") == 1:
        device = data.get("device", "(unknown)")
        print(f"🔥 불꽃 감지됨! [디바이스: {device}]")
        send_fcm_notification_to_all("불꽃 감지", f"🔥 {device} 장치에서 불꽃이 감지되었습니다.")
    return jsonify({"received": True})



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

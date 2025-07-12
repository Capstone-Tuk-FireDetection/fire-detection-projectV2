from flask import Flask, jsonify, request, Response, stream_with_context
import requests
import firebase_admin
from firebase_admin import credentials, auth, messaging, db
from functools import wraps
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Firebase 초기화 (DB URL 추가)
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://firedetection-3d3d5-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# ✅ 메모리 저장소 (FCM 토큰만 남김)
fcm_tokens = []

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

    if token not in fcm_tokens:
        fcm_tokens.append(token)
    print(f"🔔 공개 등록��� FCM 토큰: {token[:16]}...")
    return jsonify({"status": "token registered (public)"})

# ✅ 디바이스 등록 (Firebase DB 사용)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    ip = data.get("ip")
    name = data.get("device_name")
    if not ip or not name:
        return jsonify({"error": "Invalid payload"}), 400

    ref = db.reference(f'devices/{name}')
    ref.set({
        'ip': ip,
        'status': 'offline'  # 초기 상태는 offline
    })
    return jsonify({"status": "ok", "device_name": name, "device_ip": ip})

# ✅ 디바이스 상태 업데이트 (Firebase DB 사용)
@app.route("/device_status", methods=["POST"])
def device_status():
    data = request.json
    device_name = data.get("device_name")
    status = data.get("status")
    if not device_name or status not in ["online", "offline"]:
        return jsonify({"error": "Invalid payload"}), 400

    ref = db.reference(f'devices/{device_name}/status')
    ref.set(status)
    print(f"✅ DB 상태 업데이트: {device_name} is {status}")
    return jsonify({"status": "ok"})

# ✅ 디바이스 목록 조회 (Firebase DB 사용)
@app.route("/devices")
def list_devices():
    ref = db.reference('devices')
    devices = ref.get()
    return jsonify(devices if devices else {})

# ✅ 사용자 디바이스 등록 (Firebase DB 사용)
@app.route("/user/devices", methods=["POST"])
@firebase_required
def register_user_device():
    data = request.json
    device_name = data.get("device_name")
    
    # devices 경로에서 ip를 한번 더 확인
    device_info = db.reference(f'devices/{device_name}').get()
    if not device_info or 'ip' not in device_info:
        return jsonify({"error": "Device not found in global list"}), 404
    
    ip = device_info['ip']
    
    ref = db.reference(f'user_devices/{request.uid}/{device_name}')
    ref.set({'ip': ip}) # ip 정보와 함께 저장
    return jsonify({"status": "registered", "device_name": device_name})

# ✅ 사용자 디바이스 목록 (Firebase DB 사용)
@app.route("/user/devices", methods=["GET"])
@firebase_required
def get_user_devices():
    ref = db.reference(f'user_devices/{request.uid}')
    user_devices = ref.get()
    return jsonify(user_devices if user_devices else {})

# ✅ 사용자 디바이스 삭제 (Firebase DB 사용)
@app.route("/user/devices/<device_name>", methods=["DELETE"])
@firebase_required
def delete_user_device(device_name):
    ref = db.reference(f'user_devices/{request.uid}/{device_name}')
    if ref.get():
        ref.delete()
        return jsonify({"status": "deleted", "device_name": device_name})
    return jsonify({"error": "Device not found"}), 404

# ✅ flame 상태 조회
@app.route("/flame/<device>")
def get_flame_by_name(device):
    device_info = db.reference(f'devices/{device}').get()
    if not device_info or 'ip' not in device_info:
        return jsonify({"flame": -1, "error": "Device not found"}), 404
    
    ip = device_info['ip']
    try:
        resp = requests.get(f"http://{ip}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503

# 디바이스별 스트리밍 요청 상태 캐싱 (스레드 안전하게)
active_stream_clients = {}
lock = threading.Lock()

@app.route("/stream/<device>")
def stream_device(device):
    device_info = db.reference(f'devices/{device}').get()
    if not device_info or 'ip' not in device_info:
        return Response("Device not found", status=404)
    
    ip = device_info['ip']

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


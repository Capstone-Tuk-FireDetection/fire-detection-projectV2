# flask_server_combined.py
from flask import Flask, jsonify, Response, stream_with_context, request
import requests
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
import socket
import time

app = Flask(__name__)

# ✅ Firebase Admin 초기화
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# ✅ 메모리 기반 저장소
registered_devices = {}      # 공개용 디바이스 (mDNS 기반)
user_devices = {}            # 사용자별 디바이스 (인증 필요)

# ✅ mDNS lookup 기능 통합
def resolve_mdns(name, timeout=3):
    class MDNSListener(ServiceListener):
        def __init__(self):
            self.address = None
        def add_service(self, zeroconf, type, name):
            info = zeroconf.get_service_info(type, name)
            if info:
                addr = socket.inet_ntoa(info.addresses[0])
                self.address = addr

    zeroconf = Zeroconf()
    listener = MDNSListener()
    ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    for _ in range(timeout * 10):
        if listener.address:
            zeroconf.close()
            return listener.address
        time.sleep(0.1)
    zeroconf.close()
    return None

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

# ✅ 공개 디바이스 등록 (ESP32에서 호출)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("device_name")
    ip = data.get("ip")
    if name and ip:
        registered_devices[name] = ip
        return jsonify({"status": "ok", "device_name": name})
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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

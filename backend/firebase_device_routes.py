# firebase_device_routes.py
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps

app = Flask(__name__)

# ✅ Firebase Admin SDK 초기화
cred = credentials.Certificate("firebase-adminsdk.json")
firebase_admin.initialize_app(cred)

# ✅ 사용자 디바이스 저장소 (메모리용)
user_devices = {}  # { uid: { device_name: ip } }

# ✅ Firebase ID 토큰 검증 데코레이터
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

# ✅ 디바이스 등록
@app.route("/user/devices", methods=["POST"])
@firebase_required
def register_device():
    data = request.json
    device_name = data.get("device_name")
    ip = data.get("ip")
    if not device_name or not ip:
        return jsonify({"error": "device_name and ip required"}), 400

    user_devices.setdefault(request.uid, {})[device_name] = ip
    return jsonify({"status": "registered", "device_name": device_name})

# ✅ 디바이스 조회
@app.route("/user/devices", methods=["GET"])
@firebase_required
def list_user_devices():
    return jsonify(user_devices.get(request.uid, {}))

# ✅ 디바이스 삭제
@app.route("/user/devices/<device_name>", methods=["DELETE"])
@firebase_required
def delete_user_device(device_name):
    devices = user_devices.get(request.uid, {})
    if device_name in devices:
        del devices[device_name]
        return jsonify({"status": "deleted", "device_name": device_name})
    return jsonify({"error": "Device not found"}), 404

if __name__ == "__main__":
    app.run(port=8080, debug=True)

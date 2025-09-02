from flask import Flask, jsonify, request, Response, stream_with_context
import requests
import firebase_admin
from firebase_admin import credentials, auth, messaging, firestore
from functools import wraps
from flask_cors import CORS
import threading
import socket
import ipaddress
import time
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ✅ Firebase 초기화 (절대 경로 사용으로 수정)
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "firebase-adminsdk.json")
cred = credentials.Certificate(json_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ✅ 메모리 저장소 (스트리밍 클라이언트 상태만 남김)
active_stream_clients = {}
lock = threading.Lock()

# --- 🔽 [추가] 네트워크 탐색 기능 🔽 ---

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def probe_device(ip, server_ip):
    try:
        url = f"http://{ip}/discovery"
        payload = {"server_ip": server_ip}
        requests.post(url, json=payload, timeout=0.5)
        print(f"📡 Discovery: {ip}에 탐색 신호 전송 완료.")
    except requests.RequestException:
        pass
    except Exception as e:
        print(f"❌ Discovery: {ip} 탐색 중 오류: {e}")

def network_scanner(server_ip):
    print(f"💡 [Auto-Discovery] 서버 IP {server_ip} 기준으로 장치 탐색을 시작합니다...")
    try:
        interface = ipaddress.ip_interface(f"{server_ip}/24")
        subnet = interface.network
        print(f"💡 [Auto-Discovery] 스캔 대상 서브넷: {subnet}")
    except ValueError:
        print(f"❌ [Auto-Discovery] 서브넷을 확인할 수 없어 탐색을 중단합니다.")
        return

    threads = []
    for ip in subnet.hosts():
        ip_str = str(ip)
        if ip_str == server_ip:
            continue

        thread = threading.Thread(target=probe_device, args=(ip_str, server_ip))
        threads.append(thread)
        thread.start()
        
        if len(threads) % 20 == 0:
            time.sleep(0.1)

    for thread in threads:
        thread.join()

    print("✅ [Auto-Discovery] 네트워크 장치 탐색 완료.")

@app.route("/rescan_devices", methods=["POST"])
def rescan_devices():
    server_ip = get_local_ip()
    scanner_thread = threading.Thread(target=network_scanner, args=(server_ip,))
    scanner_thread.start()
    return jsonify({"status": "rescan started"}), 202

# --- 🔼 [추가] 네트워크 탐색 기능 🔼 ---


# ✅ FCM 알림 함수 (Firestore 연동으로 수정)
def send_fcm_notification_to_all(title, body):
    tokens_to_delete = []
    
    # Firestore에서 모든 FCM 토큰 문서를 가져옴
    docs = db.collection('fcm_tokens').stream()

    for doc in docs:
        token = doc.id
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token
        )
        try:
            response = messaging.send(message)
            print(f"✅ FCM 전송 완료: {token[:16]}... → {response}")
        except messaging.UnregisteredError:
            print(f"🗑️ 등록되지 않은 토큰 발견, 삭제 예정: {token[:16]}...")
            tokens_to_delete.append(token)
        except Exception as e:
            print(f"❌ FCM 전송 실패: {token[:16]}... → {e}")

    # Firestore에서 유효하지 않은 토큰들 삭제
    if tokens_to_delete:
        print(f"--- 유효하지 않은 FCM 토큰 {len(tokens_to_delete)}개 삭제 시작 ---")
        for token in tokens_to_delete:
            db.collection('fcm_tokens').document(token).delete()
        print("--- 토큰 삭제 완료 ---")

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

# ✅ FCM 토큰 등록 (Firestore 연동으로 수정)
@app.route("/register_token", methods=["POST"])
def register_fcm_token():
    data = request.json
    token = data.get("token")
    if not token:
        return jsonify({"error": "FCM token required"}), 400

    # 토큰을 문서 ID로 사용하여 Firestore에 저장 (중복 방지)
    doc_ref = db.collection('fcm_tokens').document(token)
    doc_ref.set({
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    
    print(f"🔔 Firestore에 FCM 토큰 등록: {token[:16]}...")
    return jsonify({"status": "token registered in Firestore"})

# ✅ 디바이스 등록 (Firestore 사용)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    ip = data.get("ip")
    name = data.get("device_name")
    if not ip or not name:
        return jsonify({"error": "Invalid payload"}), 400

    doc_ref = db.collection('devices').document(name)
    doc_ref.set({
        'ip': ip,
        'status': 'offline'
    })
    print(f"✅ Device Registered: {name} at {ip}")
    return jsonify({"status": "ok", "device_name": name, "device_ip": ip})

# ... (이하 다른 API 핸들러들은 변경 없음) ...

# ✅ 디바이스 상태 업데이트 (Firestore 사용)
@app.route("/device_status", methods=["POST"])
def device_status():
    data = request.json
    device_name = data.get("device_name")
    status = data.get("status")
    if not device_name or status not in ["online", "offline"]:
        return jsonify({"error": "Invalid payload"}), 400

    doc_ref = db.collection('devices').document(device_name)
    doc_ref.update({'status': status})
    print(f"✅ DB 상태 업데이트: {device_name} is {status}")
    return jsonify({"status": "ok"})

# ✅ 디바이스 목록 조회 (Firestore 사용)
@app.route("/devices")
def list_devices():
    docs = db.collection('devices').stream()
    devices = {doc.id: doc.to_dict() for doc in docs}
    return jsonify(devices if devices else {})

# ✅ 사용자 디바이스 등록 (Firestore 사용)
@app.route("/user/devices", methods=["POST"])
@firebase_required
def register_user_device():
    data = request.json
    device_name = data.get("device_name")
    
    device_doc = db.collection('devices').document(device_name).get()
    if not device_doc.exists:
        return jsonify({"error": "Device not found in global list"}), 404
    
    device_info = device_doc.to_dict()
    ip = device_info.get('ip')
    if not ip:
        return jsonify({"error": "Device IP not found in global list"}), 404

    ref = db.collection('users').document(request.uid).collection('devices').document(device_name)
    ref.set({'ip': ip})
    return jsonify({"status": "registered", "device_name": device_name})

# ✅ 사용자 디바이스 목록 (Firestore 사용)
@app.route("/user/devices", methods=["GET"])
@firebase_required
def get_user_devices():
    docs = db.collection('users').document(request.uid).collection('devices').stream()
    user_devices = {doc.id: doc.to_dict() for doc in docs}
    return jsonify(user_devices if user_devices else {})

# ✅ 사용자 디바이스 삭제 (Firestore 사용)
@app.route("/user/devices/<device_name>", methods=["DELETE"])
@firebase_required
def delete_user_device(device_name):
    doc_ref = db.collection('users').document(request.uid).collection('devices').document(device_name)
    if doc_ref.get().exists:
        doc_ref.delete()
        return jsonify({"status": "deleted", "device_name": device_name})
    return jsonify({"error": "Device not found"}), 404

# ✅ flame 상태 조회 (Firestore 사용)
@app.route("/flame/<device>")
def get_flame_by_name(device):
    doc = db.collection('devices').document(device).get()
    if not doc.exists:
        return jsonify({"flame": -1, "error": "Device not found"}), 404
    
    device_info = doc.to_dict()
    ip = device_info.get('ip')
    if not ip:
        return jsonify({"flame": -1, "error": "Device IP not found"}), 404

    try:
        resp = requests.get(f"http://{ip}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503

@app.route("/stream/<device>")
def stream_device(device):
    doc = db.collection('devices').document(device).get()
    if not doc.exists:
        return Response("Device not found", status=404)
    
    device_info = doc.to_dict()
    ip = device_info.get('ip')
    if not ip:
        return Response("Device IP not found", status=404)

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

# ✅ 스냅샷 API
@app.route("/snapshot/<device>")
def snapshot_device(device):
    doc = db.collection('devices').document(device).get()
    if not doc.exists:
        return Response("Device not found", status=404)
    
    device_info = doc.to_dict()
    ip = device_info.get('ip')
    if not ip:
        return Response("Device IP not found", status=404)

    try:
        # ESP32-CAM의 캡처 엔드포인트로 요청
        r = requests.get(f"http://{ip}/capture", timeout=5, stream=True)
        
        # ESP32-CAM으로부터 받은 응답 헤더를 그대로 클라이언트에 전달
        headers = [(name, value) for (name, value) in r.raw.headers.items()]
        
        # 이미지 데이터를 Response 객체로 감싸서 반환
        return Response(r.content, r.status_code, headers)
    
    except requests.RequestException as e:
        print(f"❌ 스냅샷 오류({device}):", e)
        return Response(f"Failed to get snapshot from {device}", status=503)

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
    server_ip = get_local_ip()
    print(f"🔥 Flask 서버를 IP {server_ip}에서 시작합니다.")
    
    scanner_thread = threading.Thread(target=network_scanner, args=(server_ip,))
    scanner_thread.daemon = True
    scanner_thread.start()

    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
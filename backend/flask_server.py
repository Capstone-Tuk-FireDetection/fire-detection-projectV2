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
import subprocess
import sys
# ★ 추가
from datetime import datetime, timezone
from google.cloud import firestore as gcf 
# 상단 import에 추가
from flask import send_from_directory


ai_process = None

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

# ★ 추가: 온라인 판정 TTL(초). TTL 초 동안 하트비트 없으면 offline 처리
ONLINE_TTL_SEC = 45
ALERT_IMG_DIR = os.path.join(script_dir, "alert_images")
os.makedirs(ALERT_IMG_DIR, exist_ok=True)
# 이미지 서빙 라우트 추가
@app.route("/alert_image/<name>")
def alert_image(name):
    return send_from_directory(ALERT_IMG_DIR, name, mimetype="image/jpeg", as_attachment=False)

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
    except requests.exceptions.RequestException as e:
        # Log connection errors, but only common ones to avoid spamming
        if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            # print(f"- Discovery: Failed to connect to {ip}") # Optional: for very verbose logging
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

# 온라인 상태인 esp 선택후 AI 스트림에 사용
def _pick_online_device_name():
    """Firestore에서 online으로 간주되는 장치 중 last_seen이 가장 최근인 장치명을 리턴."""
    now = datetime.now(timezone.utc)
    best_name = None
    best_seen = datetime.fromtimestamp(0, tz=timezone.utc)

    for doc in db.collection('devices').stream():
        d = doc.to_dict() or {}
        last_seen = _to_dt(d.get('last_seen'))
        if last_seen and (now - last_seen).total_seconds() <= ONLINE_TTL_SEC:
            if last_seen > best_seen:
                best_name = doc.id
                best_seen = last_seen
    return best_name


# ✅ FCM 알림 함수 (Firestore 연동으로 수정)
def send_fcm_notification_to_all(title, body):
    tokens_to_delete = []
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

    # IP 충돌 방지: 이 IP를 사용하던 다른 장치가 있으면 IP를 None으로 변경
    docs = db.collection('devices').where('ip', '==', ip).stream()
    for doc in docs:
        if doc.id != name:
            print(f"⚠️ IP conflict detected during registration. Device '{doc.id}' had the same IP {ip}. Clearing its IP.")
            doc.reference.update({'ip': None})

    doc_ref = db.collection('devices').document(name)
    # ★ 등록 즉시 online + last_seen 기록
    doc_ref.set({
        'ip': ip,
        'status': 'online',
        'last_seen': firestore.SERVER_TIMESTAMP
    }, merge=True)

    print(f"✅ Device Registered: {name} at {ip}")
    return jsonify({"status": "ok", "device_name": name, "device_ip": ip})


# ✅ 디바이스 상태 업데이트 (Firestore 사용)
@app.route("/device_status", methods=["POST"])
def device_status():
    data = request.json
    device_name = data.get("device_name")
    status = data.get("status")
    if not device_name or status not in ["online", "offline"]:
        return jsonify({"error": "Invalid payload"}), 400

    doc_ref = db.collection('devices').document(device_name)
    # ★ status 수동 업데이트 시 last_seen도 갱신(online일 경우)
    upd = {'status': status}
    if status == 'online':
        upd['last_seen'] = firestore.SERVER_TIMESTAMP
    doc_ref.update(upd)

    print(f"✅ DB 상태 업데이트: {device_name} is {status}")
    return jsonify({"status": "ok"})


# ★ 추가: 하트비트 수신
@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json() or {}
    name = data.get("device_name")
    ip = data.get("ip")
    if not name:
        return jsonify({"error": "device_name required"}), 400

    # IP 충돌 방지: 이 IP를 사용하던 다른 장치가 있으면 IP를 None으로 변경
    if ip:
        docs = db.collection('devices').where('ip', '==', ip).stream()
        for doc in docs:
            if doc.id != name:
                print(f"⚠️ IP conflict detected during heartbeat. Device '{doc.id}' had the same IP {ip}. Clearing its IP.")
                doc.reference.update({'ip': None})

    upd = {
        'status': 'online',
        'last_seen': firestore.SERVER_TIMESTAMP
    }
    if ip:
        upd['ip'] = ip

    db.collection('devices').document(name).set(upd, merge=True)
    # print(f"💓 heartbeat: {name} ({ip})")
    return jsonify({"status": "ok"})


# ★ 추가: Firestore Timestamp -> timezone-aware datetime
def _to_dt(ts):
    try:
        if hasattr(ts, "to_datetime"):
            return ts.to_datetime()
        return ts
    except Exception:
        return None

def poke_known_devices_loop(interval_sec=100):  # ★ 추가
    """Firestore에 등록된 장치 IP들에만 /discovery를 주기적으로 보내서
    재부팅 후에도 하트비트가 재개되도록 보장."""
    while True:
        global ai_process
        if ai_process and ai_process.poll() is None:
            # AI 스트림이 실행 중일 때는 장치 탐색을 일시 중지합니다.
            time.sleep(interval_sec)
            continue

        try:
            server_ip = get_local_ip()
            docs = db.collection('devices').stream()
            for d in docs:
                info = d.to_dict() or {}
                ip = info.get('ip')
                if not ip:
                    continue
                # 기존 probe_device 재사용
                probe_device(ip, server_ip)
        except Exception as e:
            print("⚠️ poke_known_devices_loop error:", e)
        time.sleep(interval_sec)

# ✅ 디바이스 목록 조회 (Firestore 사용)  — last_seen 기반 온라인 판정 적용
@app.route("/devices")
def list_devices():
    now = datetime.now(timezone.utc)
    out = {}
    for doc in db.collection('devices').stream():
        d = doc.to_dict() or {}
        last_seen = _to_dt(d.get('last_seen'))
        is_online = False
        if last_seen:
            is_online = (now - last_seen).total_seconds() <= ONLINE_TTL_SEC
        out[doc.id] = {
            "ip": d.get("ip"),
            "status": "online" if is_online else "offline",
            "last_seen": last_seen.isoformat() if last_seen else None,
            "last_offline_at": (_to_dt(d.get("last_offline_at")).isoformat()
                                if d.get("last_offline_at") else None)
        }
    return jsonify(out if out else {})


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
        data = resp.json()
        # ★ 성공적으로 응답 받으면 online 판정으로 즉시 갱신
        db.collection('devices').document(device).update({
            'last_seen': firestore.SERVER_TIMESTAMP,
            'status': 'online'
        })
        return jsonify(data)
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
            # ★ 스트림 연결 성공 == 온라인
            db.collection('devices').document(device).update({
                'last_seen': firestore.SERVER_TIMESTAMP,
                'status': 'online'
            })
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
    print(f"-> Received /snapshot request for device: {device}")
    doc = db.collection('devices').document(device).get()
    if not doc.exists:
        print(f"  - ❌ Snapshot failed: Device '{device}' not found in Firestore.")
        return Response("Device not found", status=404)
    
    device_info = doc.to_dict()
    ip = device_info.get('ip')
    print(f"  - Found device '{device}' in Firestore with IP: {ip}")
    if not ip:
        print(f"  - ❌ Snapshot failed: IP address not found for device '{device}'.")
        return Response("Device IP not found", status=404)

    try:
        print(f"  - Attempting to get snapshot from http://{ip}/jpg")
        r = requests.get(f"http://{ip}/jpg", timeout=5, stream=True)
        
        print(f"  - Snapshot request to ESP32 returned status: {r.status_code}")
        
        # ★ 성공하면 last_seen/status 갱신
        db.collection('devices').document(device).update({
            'last_seen': firestore.SERVER_TIMESTAMP,
            'status': 'online'
        })
        headers = [(name, value) for (name, value) in r.raw.headers.items()]
        return Response(r.content, r.status_code, headers)
    except requests.RequestException as e:
        print(f"  - ❌ Snapshot request to ESP32 failed: {e}")
        return Response(f"Failed to get snapshot from {device}", status=503)

def _save_alert_snapshot(device_name: str):
    """장치의 /jpg를 받아 로컬에 저장하고, 성공 시 상대 경로(/alert_image/...) 리턴."""
    try:
        doc = db.collection('devices').document(device_name).get()
        if not doc.exists:
            return None
        info = doc.to_dict() or {}
        ip = info.get('ip')
        if not ip:
            return None

        # ESP에서 바로 한 장 가져오기
        r = requests.get(f"http://{ip}/jpg", timeout=2)
        if r.status_code != 200:
            return None

        # 파일명: device_타임스탬프.jpg
        fname = f"{device_name}_{int(time.time())}.jpg"
        fpath = os.path.join(ALERT_IMG_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(r.content)

        # 프론트엔드는 backendBaseUrl + 이 경로로 접근
        return f"/alert_image/{fname}"
    except Exception as e:
        print("⚠️ snapshot save error:", e)
        return None


def _attach_image_async(alert_id: str, device_name: str):
    """비동기로 스냅샷 저장 → 성공 시 해당 알림 문서에 image_url 업데이트."""
    url = _save_alert_snapshot(device_name)
    if url:
        try:
            db.collection('alerts').document(alert_id).update({'image_url': url})
        except Exception as e:
            print("⚠️ failed to update alert with image_url:", e)


@app.route("/alert", methods=["POST"])
def alert():
    data = request.get_json()
    if data.get("flame") == 1:
        device = data.get("device", "(unknown)")
        title = "불꽃 감지"
        body = f"🔥 {device} 장치에서 불꽃이 감지되었습니다."
        print(f"🔥 불꽃 감지됨! [디바이스: {device}]")

        # 1) 알림 먼저 저장 (image_url 없이)
        doc_ref = db.collection('alerts').document()
        doc_ref.set({
            'device': device,
            'title': title,
            'body': body,
            'flame': 1,
            'created_at': firestore.SERVER_TIMESTAMP,
            # 'image_url': 나중에 비동기로 붙임
        })

        # 2) 비동기로 스냅샷 저장 후 image_url 업데이트
        threading.Thread(target=_attach_image_async, args=(doc_ref.id, device), daemon=True).start()

        # 3) FCM 발송은 기존 그대로
        send_fcm_notification_to_all(title, body)
    return jsonify({"received": True})





def _parse_iso_or_none(s):
    if not s:
        return None
    try:
        # 'Z' → '+00:00' 보정
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return None

@app.route("/alerts", methods=["GET"])
def list_alerts():
    # /alerts?limit=50&device=espcam1&since=2025-09-13T00:00:00+00:00
    limit = int(request.args.get("limit", "50"))
    device = request.args.get("device")
    since_iso = request.args.get("since")
    since_dt = _parse_iso_or_none(since_iso)

    q = db.collection('alerts')
    if device:
        q = q.where('device', '==', device)
    if since_dt:
        q = q.where('created_at', '>=', since_dt)

    # 최신순 정렬
    q = q.order_by('created_at', direction=gcf.Query.DESCENDING).limit(limit)

    items = []
    for doc in q.stream():
        d = doc.to_dict() or {}
        created = _to_dt(d.get('created_at'))
        items.append({
            "id": doc.id,
            "device": d.get("device"),
            "title": d.get("title"),
            "body": d.get("body"),
            "flame": d.get("flame"),
            "image_url": d.get("image_url"), 
            "created_at": created.isoformat() if created else None,
        })
    return jsonify(items)

@app.route("/alerts", methods=["DELETE"])
def delete_alerts():
    # /alerts?device=espcam1&limit=500
    device = request.args.get("device")
    try:
        limit = int(request.args.get("limit", "500"))  # 안전을 위해 기본 500개만
    except ValueError:
        limit = 500

    q = db.collection('alerts')
    if device:
        q = q.where('device', '==', device)

    deleted = 0
    CHUNK = 200
    while deleted < limit:
        # 정렬을 넣어 주면 페이지네이션/반복 삭제 시 더 안정적
        chunk = list(q.order_by('created_at', direction=gcf.Query.DESCENDING)
                       .limit(min(CHUNK, limit - deleted)).stream())
        if not chunk:
            break

        for snap in chunk:
            d = snap.to_dict() or {}
            image_url = (d.get('image_url') or "").strip()

            # /alert_image/<파일명> 형태만 처리
            if image_url.startswith("/alert_image/"):
                name = os.path.basename(image_url.replace("/alert_image/", "", 1))
                if name:
                    try:
                        os.remove(os.path.join(ALERT_IMG_DIR, name))
                    except FileNotFoundError:
                        # 이미 없는 경우는 무시
                        pass
                    except Exception as e:
                        print(f"⚠️ image delete failed: {name} -> {e}")

            # Firestore 문서 삭제
            try:
                snap.reference.delete()
            except Exception as e:
                print("⚠️ delete doc failed:", e)

        deleted += len(chunk)

    return jsonify({"deleted": deleted})



@app.route("/start_ai_stream", methods=["POST"])
def start_ai_stream():
    global ai_process
    if ai_process and ai_process.poll() is None:
        return jsonify({"status": "AI stream already running"}), 200

    data = request.get_json(silent=True) or {}
    req_device = (data.get("device_name") or "").strip()

    # ▶ 추가: 판독 소스 플래그 (기본값: 둘 다 True)
    use_ai = bool(data.get("use_ai", True))
    use_sensor = bool(data.get("use_sensor", True))

    # ▶ 추가: 둘 다 해제면 시작 거절
    if not (use_ai or use_sensor):
        return jsonify({"error": "at least one of use_ai/use_sensor must be true"}), 400

    # --- (기존) 온라인 장치 자동/검증 로직 그대로 ---
    if not req_device or req_device.lower() == "auto":
        chosen = _pick_online_device_name()
        if not chosen:
            return jsonify({"error": "no online devices"}), 409
    else:
        doc = db.collection('devices').document(req_device).get()
        if not doc.exists:
            return jsonify({"error": f"device '{req_device}' not found"}), 404
        d = doc.to_dict() or {}
        last_seen = _to_dt(d.get('last_seen'))
        now = datetime.now(timezone.utc)
        is_online = last_seen and (now - last_seen).total_seconds() <= ONLINE_TTL_SEC
        if not is_online:
            return jsonify({"error": f"device '{req_device}' is offline"}), 409
        chosen = req_device

    try:
        ai_script_path = os.path.join(script_dir, 'stream_flame_detection.py')
        if not os.path.exists(ai_script_path):
            return jsonify({"error": f"AI script not found at {ai_script_path}"}), 500

        server_url = f"http://{get_local_ip()}:8080"

        # ▶ 추가: 인자로 플래그 넘김 (--use_ai/--use_sensor)
        command = [
            sys.executable,
            ai_script_path,
            "--device_name", chosen,
            "--server_url", server_url,
            "--use_ai", "1" if use_ai else "0",
            "--use_sensor", "1" if use_sensor else "0",
        ]

        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE

        ai_process = subprocess.Popen(command, **popen_kwargs)
        print(f"✅ AI stream started with PID: {ai_process.pid} (device: {chosen}, use_ai={use_ai}, use_sensor={use_sensor})")
        return jsonify({"status": "AI stream started", "pid": ai_process.pid, "device_name": chosen}), 200
    except Exception as e:
        print(f"❌ Failed to start AI stream: {e}")
        return jsonify({"error": f"Failed to start AI stream: {str(e)}"}), 500



@app.route("/stop_ai_stream", methods=["POST"])
def stop_ai_stream():
    global ai_process
    if ai_process and ai_process.poll() is None:
        try:
            ai_process.terminate() # Gracefully terminate the process
            ai_process.wait(timeout=5) # Wait for it to terminate
            print(f"✅ AI stream (PID: {ai_process.pid}) stopped.")
            ai_process = None
            return jsonify({"status": "AI stream stopped"}), 200
        except subprocess.TimeoutExpired:
            ai_process.kill() # Force kill if terminate fails
            print(f"🚨 AI stream (PID: {ai_process.pid}) killed after timeout.")
            ai_process = None
            return jsonify({"status": "AI stream killed (timeout)"}), 200
        except Exception as e:
            print(f"❌ Error stopping AI stream: {e}")
            return jsonify({"error": f"Error stopping AI stream: {str(e)}"}), 500
    else:
        return jsonify({"status": "AI stream not running"}), 200

@app.route("/ai_stream_status", methods=["GET"])
def ai_stream_status():
    global ai_process
    if ai_process and ai_process.poll() is None:
        return jsonify({"status": "running", "pid": ai_process.pid}), 200
    else:
        return jsonify({"status": "not running"}), 200
# --- End AI Stream Control Endpoints ---


# ★ 추가: 오래된(last_seen) 장치 offline 스윕
def offline_sweeper():
    while True:
        try:
            now = datetime.now(timezone.utc)
            for doc in db.collection('devices').stream():
                d = doc.to_dict() or {}
                last_seen = _to_dt(d.get('last_seen'))
                if not last_seen:
                    continue
                age = (now - last_seen).total_seconds()
                if age > ONLINE_TTL_SEC and d.get('status') != 'offline':
                    doc.reference.update({
                        'status': 'offline',
                        'last_offline_at': firestore.SERVER_TIMESTAMP
                    })
        except Exception as e:
            print("⚠️ offline_sweeper error:", e)
        time.sleep(15)  # 15초 간격 확인


if __name__ == '__main__':
    server_ip = get_local_ip()
    print(f"🔥 Flask 서버를 IP {server_ip}에서 시작합니다.")

    threading.Thread(target=network_scanner, args=(server_ip,), daemon=True).start()
    threading.Thread(target=offline_sweeper, daemon=True).start()
    threading.Thread(target=poke_known_devices_loop, daemon=True).start()  # ★ 추가

    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)


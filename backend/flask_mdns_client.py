# flask_mdns_client.py
from flask import Flask, jsonify, Response, stream_with_context, request
import requests
from mdns_lookup import resolve_mdns

app = Flask(__name__)

# 등록된 디바이스 저장소 (메모리용)
registered_devices = {}

# mDNS 이름을 실제 IP로 변환
resolved_ip = resolve_mdns("espcam2")
if resolved_ip:
    CAMERA_HOST = f"http://{resolved_ip}"
else:
    print("❗ mDNS 이름을 IP로 해석하지 못했습니다.")
    CAMERA_HOST = "http://espcam2.local"

@app.route("/flame")
def get_flame():
    try:
        resp = requests.get(f"{CAMERA_HOST}/flame", timeout=2)
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"flame": -1, "error": str(e)}), 503

@app.route("/stream")
def stream():
    try:
        r = requests.get(f"{CAMERA_HOST}/stream", stream=True, timeout=5)
        return Response(
            stream_with_context(r.iter_content(chunk_size=1024)),
            content_type=r.headers.get("Content-Type", "multipart/x-mixed-replace")
        )
    except requests.RequestException as e:
        return Response(f"Stream error: {str(e)}", status=503)

# ✅ 디바이스 등록 요청 처리
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    device_name = data.get("device_name")
    ip = data.get("ip")
    if device_name and ip:
        registered_devices[device_name] = ip
        return jsonify({"status": "ok", "registered": device_name})
    return jsonify({"status": "error", "message": "Invalid payload"}), 400

# ✅ 등록된 디바이스 목록 조회
@app.route("/devices")
def list_devices():
    return jsonify(registered_devices)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

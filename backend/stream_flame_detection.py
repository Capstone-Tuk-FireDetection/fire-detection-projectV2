import requests
import argparse
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# 🔥 AI 모델 정의
class FlameClassifier(nn.Module):
    def __init__(self):
        super(FlameClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (64 // 4) * (64 // 4), 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 📨 Flask 서버로 알림 전송
def send_alert_to_flask(server_url, device_name=None, ip=None):
    payload = {"flame": 1}
    if device_name:
        payload["device"] = device_name
    elif ip:
        payload["device"] = ip  # IP라도 전달
    try:
        response = requests.post(f"{server_url}/alert", json=payload)
        print("알림 전송:", response.status_code, response.text)
    except Exception as e:
        print("Flask 전송 실패:", e)

# 🔎 불꽃 센서 값 읽기
def read_flame_sensor(ip):
    try:
        response = requests.get(f"http://{ip}/flame", timeout=1)
        if response.ok:
            return response.json().get("flame", -1)
    except:
        pass
    return -1

# 🔍 실시간 AI 추론 루프
def run_inference(ip, server_url):
    stream_url = f"http://{ip}/stream"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f'스트림 열기 실패: {stream_url}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlameClassifier().to(device)
    model.load_state_dict(torch.load('./flame_cnn.pth', map_location=device))
    model.eval()

    print(f"[{ip}] 스트리밍 시작, 불꽃 감지 중 (AND 조건)...")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('프레임 읽기 실패')
                break

            # 🔍 AI 추론
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                flame_prob = torch.softmax(output, dim=1)[0][0].item()

            # 🔥 불꽃 센서 값
            sensor_value = read_flame_sensor(ip)

            # ✅ AND 조건: AI + 센서 모두 감지해야 알림
            threshold = 0.85
            ai_detected = flame_prob > threshold
            sensor_detected = (sensor_value == 1)
            final_result = ai_detected and sensor_detected

            if final_result:
                send_alert_to_flask(server_url, ip=ip)

            # 디버깅용 표시
            status = "🔥 FLAME DETECTED" if final_result else f"AI:{ai_detected} / SENSOR:{sensor_detected}"
            color = (0, 0, 255) if final_result else (200, 200, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except KeyboardInterrupt:
        print("Ctrl+C 종료")

    finally:
        cap.release()
        print("종료됨")

# 🏁 진입점
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="ESP32-CAM IP 주소")
    parser.add_argument("--server_url", default="http://localhost:8080", help="Flask 서버 주소")
    args = parser.parse_args()

    run_inference(args.ip, args.server_url)

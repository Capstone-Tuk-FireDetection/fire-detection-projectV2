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

def send_alert_to_flask(server_url, device_name=None):
    payload = {"flame": 1}
    if device_name:
        payload["device"] = device_name
    try:
        response = requests.post(f"{server_url}/alert", json=payload)
        print("알림 전송:", response.status_code, response.text)
    except Exception as e:
        print("Flask 전송 실패:", e)

def read_flame_sensor(server_url, device_name):
    try:
        response = requests.get(f"{server_url}/flame/{device_name}", timeout=1)
        if response.ok:
            return response.json().get("flame", -1)
    except:
        pass
    return -1

def run_inference(device_name, server_url):
    stream_url = f"{server_url}/stream/{device_name}"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f'스트림 열기 실패: {stream_url}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlameClassifier().to(device)
    model.load_state_dict(torch.load('./flame_cnn.pth', map_location=device))
    model.eval()

    print(f"[{device_name}] 스트리밍 시작 (프록시 통해 분석 중)...")

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

            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                flame_prob = torch.softmax(output, dim=1)[0][0].item()

            sensor_value = read_flame_sensor(server_url, device_name)

            ai_detected = flame_prob > 0.85
            sensor_detected = (sensor_value == 1)
            final_result = ai_detected and sensor_detected

            if final_result:
                send_alert_to_flask(server_url, device_name)

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
    parser.add_argument("--device_name", required=True, help="Flask에 등록된 디바이스 이름")
    parser.add_argument("--server_url", default="http://localhost:8080", help="Flask 서버 주소")
    args = parser.parse_args()

    run_inference(args.device_name, args.server_url)
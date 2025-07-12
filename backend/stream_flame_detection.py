import requests
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import time

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

# 👈 [추가] 디바이스 상태 업데이트 함수
def update_device_status(server_url, device_name, status):
    try:
        payload = {"device_name": device_name, "status": status}
        requests.post(f"{server_url}/device_status", json=payload, timeout=2)
        print(f"[{device_name}] 상태 업데이트 전송: {status}")
    except Exception as e:
        print(f"[{device_name}] 상태 업데이트 실패: {e}")


def read_flame_sensor(server_url, device_name):
    try:
        response = requests.get(f"{server_url}/flame/{device_name}", timeout=1)
        if response.ok:
            return response.json().get("flame", -1)
    except:
        pass
    return -1

def run_inference(device_name, server_url):
    # 👈 [수정] registered_devices에서 ip만 추출
    device_ip = registered_devices.get(device_name, {}).get("ip")
    if not device_ip:
        print(f"❌ '{device_name}'의 IP를 찾을 수 없습니다.")
        return

    jpg_url = f"http://{device_ip}/jpg"
    print(f"[{device_name}] JPEG 기반 추론 시작: {jpg_url}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlameClassifier().to(device)
    model.load_state_dict(torch.load('./flame_cnn.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    last_alert_time = 0
    alert_interval = 30  # 초당 알림 제한
    
    # 👈 [추가] 연속 실패 카운터
    failure_count = 0
    FAILURE_THRESHOLD = 5 # 5회 연속 실패 시 오프라인 간주

    # 👈 [수정] try...finally 구문으로 감싸기
    try:
        # 👈 [추가] 시작 시 online 상태 보고
        update_device_status(server_url, device_name, "online")

        while True:
            try:
                r = requests.get(jpg_url, timeout=2)
                r.raise_for_status() # 200 OK 아니면 예외 발생
                np_arr = np.frombuffer(r.content, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("프레임 디코딩 실패")
                
                failure_count = 0 # 👈 [추가] 성공 시 카운터 초기화

            except Exception as e:
                failure_count += 1 # 👈 [추가] 실패 시 카운터 증가
                print(f"❌ [{device_name}] 요청 실패 ({failure_count}/{FAILURE_THRESHOLD}): {e}")
                if failure_count >= FAILURE_THRESHOLD:
                    print(f"🚨 [{device_name}] 오프라인으로 간주. 프로세스를 종료합니다.")
                    break # 루프 탈출
                time.sleep(2) # 실패 시 잠시 대기
                continue

            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                flame_prob = torch.softmax(output, dim=1)[0][0].item()

            sensor_value = read_flame_sensor(server_url, device_name)
            ai_detected = flame_prob > 0.85
            sensor_detected = (sensor_value == 1)
            final_result = ai_detected and sensor_detected

            # 알림 제한
            current_time = time.time()
            if final_result and (current_time - last_alert_time > alert_interval):
                send_alert_to_flask(server_url, device_name)
                last_alert_time = current_time

            # 디버깅 표시
            status = "🔥 DETECTED" if final_result else f"AI:{ai_detected} / SENSOR:{sensor_detected}"
            color = (0, 0, 255) if final_result else (150, 150, 150)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow(device_name, frame)
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print(f"종료 (Ctrl+C) [{device_name}]")

    finally:
        # 👈 [추가] 종료 시 offline 상태 보고
        update_device_status(server_url, device_name, "offline")
        cv2.destroyAllWindows()

# 🏁 진입점
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_name", required=True)
    parser.add_argument("--server_url", default="http://localhost:8080")
    args = parser.parse_args()

    # Flask에서 등록된 디바이스 → IP 가져오기
    try:
        response = requests.get(f"{args.server_url}/devices")
        registered_devices = response.json()
    except:
        raise RuntimeError("❌ Flask 서버에서 디바이스 목록을 불러올 수 없습니다.")

    run_inference(args.device_name, args.server_url)

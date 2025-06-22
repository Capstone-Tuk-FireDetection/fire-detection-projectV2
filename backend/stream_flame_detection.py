
import requests

def send_alert_to_flask(result):
    try:
        response = requests.post(
            "http://espcam2.local:8080/alert",
            json={"flame": result}
        )
        print("알림 전송:", response.status_code, response.text)
    except Exception as e:
        print("Flask 전송 실패:", e)

def read_flame_sensor():
    try:
        response = requests.get("http://espcam2.local/flame", timeout=1)
        if response.ok:
            return response.json().get("flame", -1)
    except:
        pass
    return -1

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from datetime import datetime

# 모델 클래스 정의 (학습할 때와 같아야 함)
class FlameClassifier(nn.Module):
    def __init__(self):
        super(FlameClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (64 // 4) * (64 // 4), 128)
        self.fc2 = nn.Linear(128, 2)  # flame / no_flame

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = FlameClassifier().to(device)
model.load_state_dict(torch.load('flame_cnn.pth', map_location=device))
model.eval()

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 스트리밍 URL
stream_url = "http://espcam2.local/stream"

# VideoCapture 열기
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    raise RuntimeError('스트림 열기 실패: ' + stream_url)

print("스트리밍 시작, 불꽃 감지 중...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('프레임 읽기 실패, 종료')
            break

        # 프레임 전처리
        input_frame = transform(frame).unsqueeze(0).to(device)

        # 불꽃 판별
        with torch.no_grad():
            outputs = model(input_frame)
            probabilities = torch.softmax(outputs, dim=1)
            flame_prob = probabilities[0][0].item()  # 클래스 0번이 flame

        # 감도 임계값 설정
        threshold = 0.85  # 85% 이상일 때만 flame으로 판단

        if flame_prob > threshold:
            text = f"Flame Detected ({flame_prob:.2f})"
            color = (0, 0, 255)
        else:
            text = f"No Flame ({flame_prob:.2f})"
            color = (0, 255, 0)

        # 프레임에 텍스트 출력
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

except KeyboardInterrupt:
    print("Ctrl+C로 종료")

finally:
    cap.release()
    print("리소스 해제 완료")
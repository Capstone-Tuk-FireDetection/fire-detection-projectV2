다음은 `stream_flame_detection.py`를 중심으로 한 프로젝트용 `README.md` 예시입니다. Flask 서버와 AI 스트리밍이 함께 작동하는 시스템을 고려해 작성하였습니다:

---

````markdown
# 🔥 ESP32-CAM Flame Detection System

본 프로젝트는 **ESP32-CAM**으로부터 실시간 영상을 받아, PyTorch 기반 인공지능 모델을 이용해 **화재(불꽃)** 여부를 탐지하고, Flask 서버를 통해 상태를 모니터링할 수 있는 IoT 시스템입니다.

---

## 📁 구성 파일

| 파일명 | 설명 |
|--------|------|
| `flask_server.py` | 사용자와 통신하는 웹 서버 (회원 관리, 디바이스 관리 등 포함) |
| `stream_flame_detection.py` | ESP32-CAM 영상 스트림을 불꽃 AI 모델로 분석 |
| `model.pth` | 사전 학습된 PyTorch 불꽃 탐지 모델 |
| `requirements.txt` | 프로젝트 실행에 필요한 Python 패키지 목록 |
| `README.md` | 프로젝트 문서 |

---

## ⚙️ 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
````

### 2. Flask 서버 실행 (1개만)

```bash
python flask_server.py
```

### 3. 스트리밍 AI 프로세스 실행 (ESP32-CAM 기기 수만큼)

```bash
python stream_flame_detection.py --camera_url http://<ESP32-CAM-IP>/stream
```

**예시:**

```bash
python stream_flame_detection.py --camera_url http://192.168.0.101/stream
```

---

## 🧠 모델 정보

* 모델: CNN 기반 flame detection 모델
* 입력: 64x64 RGB 이미지 (자동 전처리)
* 출력: `flame` 또는 `no_flame` 분류
* 저장 위치: `model.pth`

---

## 📡 시스템 구조

```plaintext
[ESP32-CAM] → [stream_flame_detection.py (AI)] → [Flask Server] → [Flutter/Web 클라이언트]
```

---

## 🔐 사용자 기능 (Flask 서버)

* Firebase 연동 로그인/회원가입
* 기기 등록/삭제
* 실시간 flame 이벤트 처리 및 저장 예정

---

## 📌 주의 사항

* `flask_server.py`는 **한 번만 실행**합니다.
* `stream_flame_detection.py`는 **기기 수만큼 병렬 실행** 가능합니다.
* 각 ESP32-CAM은 MJPEG 스트림을 제공해야 합니다.

---

## 📝 향후 개선사항

* FCM 연동 → 불꽃 감지 시 알림 전송
* 영상 저장 기능 (optional)
* 정확도 향상된 모델 교체 및 재학습 자동화

---

## 🧑‍💻 개발자

한승주
IoT 시스템 통합 및 AI 기반 영상 처리 설계자
인성준
프론트 앤드 개발 및 AI 데이터 수집
김민상
AI 데이터 수집

---

## 📜 라이선스

본 프로젝트는 MIT License 하에 공개됩니다.

```

---

필요에 따라 **Flutter 앱 설명**이나 **Firebase 설정법**, 또는 **모델 학습 코드 (`train_flame_cnn.py`)**에 대한 설명도 추가할 수 있습니다. 
```

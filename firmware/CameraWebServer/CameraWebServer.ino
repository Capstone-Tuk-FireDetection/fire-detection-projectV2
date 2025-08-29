#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#define FLAME_PIN 14 // Flame sensor 핀

// --- 🔽 [추가/수정] 전역 변수 🔽 ---
String server_ip; // 서버 IP를 저장할 변수
volatile bool is_registered = false; // 등록 상태 플래그
volatile bool allowStreaming = true;
int cachedFlame = -1;
// --- 🔼 [추가/수정] 전역 변수 🔼 ---

void sensorTask(void *param) {
  for (;;) {
    allowStreaming = false;
    delay(120);

    cachedFlame = digitalRead(FLAME_PIN);

    allowStreaming = true;
    vTaskDelay(pdMS_TO_TICKS(3000));
  }
}

#include "wifi_config.h"

// --- 🔽 [수정] registerDevice 함수 🔽 ---
void registerDevice() {
  if (server_ip.length() == 0) {
    return; // 서버 IP가 없으면 실행하지 않음
  }
  if (is_registered) {
    return; // 이미 등록되었으면 실행하지 않음
  }

  HTTPClient http;
  String url = "http://" + server_ip + "/register";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"device_name\":\"espcam1\",
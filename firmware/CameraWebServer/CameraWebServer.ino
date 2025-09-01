#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#define FLAME_PIN 14 // Flame sensor 핀

// --- 🔽 [수정] 전역 변수 🔽 ---
String server_ip; // 서버 IP를 저장할 변수
volatile bool is_registered = false; // 등록 상태 플래그
volatile bool allowStreaming = true;
int cachedFlame = -1;
// --- 🔼 [수정] 전역 변수 🔼 ---

// --- 🔽 [추가] app_httpd.cpp에서 호출될 함수 🔽 ---
extern "C" void onServerIpFound(const char* ip) {
  if (!is_registered && server_ip.length() == 0) { // 아직 등록되지 않았고, IP가 비어있을 때만 설정
    Serial.printf("✅ Server IP discovered: %s\n", ip);
    server_ip = String(ip);
  }
}
// --- 🔼 [추가] app_httpd.cpp에서 호출될 함수 🔼 ---

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

// --- 🔽 [추가] 함수 전방 선언 🔽 ---
void startCameraServer();
void registerDevice();
// --- 🔼 [추가] 함수 전방 선언 🔼 ---

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  pinMode(FLAME_PIN, INPUT);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  // --- 🔽 [수정] 카메라 설정 오타 수정 🔽 ---
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  // --- 🔼 [수정] 카메라 설정 오타 수정 🔼 ---
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QQVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 14;
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Camera init failed");
    ESP.restart();
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected");
  Serial.print("   IP Address: ");
  Serial.println(WiFi.localIP());

  // 서버가 discovery 요청을 보낼 수 있도록 웹 서버를 먼저 시작
  startCameraServer();
  Serial.println("✅ HTTP server started. Waiting for server discovery...");

  xTaskCreatePinnedToCore(sensorTask, "Sensor Task", 2048, NULL, 1, NULL, 1);
}

// --- 🔽 [수정] 장치 등록 함수 🔽 ---
void registerDevice() {
  if (server_ip.length() == 0) {
    return; // 서버 IP가 없으면 등록 시도 안 함
  }

  HTTPClient http;
  String url = "http://" + server_ip + ":8080/register";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");

  String payload = "{\"device_name\":\"espcam1\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";

  Serial.printf("🚀 Registering device to %s\n", url.c_str());
  int httpCode = http.POST(payload);

  if (httpCode == 200) {
    Serial.println("✅ Device registered successfully!");
    is_registered = true;
  } else {
    Serial.printf("❌ Device registration failed. HTTP Code: %d\n", httpCode);
    String response = http.getString();
    Serial.println("   Server response: " + response);
  }
  http.end();
}
// --- 🔼 [수정] 장치 등록 함수 🔼 ---

// --- 🔽 [수정] 메인 루프 🔽 ---
void loop() {
  if (!is_registered) {
    // 서버 IP를 수신했고, 아직 등록되지 않았다면 등록 시도
    if (server_ip.length() > 0) {
      registerDevice();
    }
    // 5초 대기 후 재시도
    delay(5000);
  } else {
    // 등록 완료 후에는 10초마다 대기
    delay(10000);
  }
}
// --- 🔼 [수정] 메인 루프 🔼 ---

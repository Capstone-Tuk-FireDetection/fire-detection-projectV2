#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>  // ✅ mDNS 사용을 위한 라이브러리 추가

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

#define FLAME_PIN 14 // Flame sensor 핀

volatile bool allowStreaming = true;
int cachedFlame = -1;

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

const char *ssid = WIFI_SSID;
const char *password = WIFI_PASSWORD;

void startCameraServer();
void setupLedFlash(int pin);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

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
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
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
    Serial.println("Camera init failed");
    return;
  }

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // ✅ mDNS 서비스 시작
  if (MDNS.begin("espcam1")) {
    Serial.println("mDNS responder started as 'espcam1.local'");
  } else {
    Serial.println("mDNS responder failed to start");
  }

  startCameraServer();
  xTaskCreatePinnedToCore(sensorTask, "Sensor Task", 2048, NULL, 1, NULL, 1);

  Serial.print("Camera Ready! Open http://");
  Serial.print(WiFi.localIP());
  Serial.println(" or http://espcam1.local to connect");
}

void loop() {
  delay(10);
}

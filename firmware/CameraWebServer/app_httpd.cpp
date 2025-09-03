// app_httpd.cpp (최신 버전, mDNS 대응 포함 + ★ heartbeat/ discovery 추가)
#include "esp_http_server.h"
#include "esp_camera.h"
#include "Arduino.h"
// ★ 추가
#include <WiFi.h>
#include <HTTPClient.h>

extern int cachedFlame;
extern volatile bool allowStreaming;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// ★ 추가: 서버 IP/디바이스 이름/하트비트 주기
static String g_serverIp = "";                // /discovery 수신 시 저장
static const char* DEVICE_NAME = "espcam1";   // 필요 시 프로젝트 설정에서 바꾸세요
static const uint32_t HEARTBEAT_MS = 10000;   // 10초 간격

// ---------- 공통 유틸: 간단 POST JSON ----------
static void post_json(const String& url, const String& json) {
  if (!WiFi.isConnected()) return;
  HTTPClient http;
  if (!http.begin(url)) return;
  http.addHeader("Content-Type", "application/json");
  http.POST((uint8_t*)json.c_str(), json.length());
  http.end();
}

static void send_register_once() {
  if (g_serverIp.isEmpty()) return;
  String url = "http://" + g_serverIp + ":8080/register";
  String body = String("{\"device_name\":\"") + DEVICE_NAME + "\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  post_json(url, body);
}

static void send_heartbeat() {
  if (g_serverIp.isEmpty()) return;
  String url = "http://" + g_serverIp + ":8080/heartbeat";
  String body = String("{\"device_name\":\"") + DEVICE_NAME + "\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  post_json(url, body);
}

// 하트비트 태스크
static void heartbeat_task(void* pv) {
  // 등록 1회 전송 후 주기적 하트비트
  send_register_once();
  for (;;) {
    send_heartbeat();
    vTaskDelay(pdMS_TO_TICKS(HEARTBEAT_MS));
  }
}

// ---------- 기존 핸들러들 ----------
static esp_err_t jpg_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ 카메라 캡처 실패 (/jpg)");
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t stream_handler(httpd_req_t *req) {
  while (!allowStreaming) {
    vTaskDelay(pdMS_TO_TICKS(100));
  }

  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    if (!allowStreaming) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    if (res == ESP_OK) {
      char header_buf[64];
      int header_len = snprintf(header_buf, sizeof(header_buf),
        "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
      res = httpd_resp_send_chunk(req, header_buf, header_len);
    }

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }

    esp_camera_fb_return(fb);

    if (res != ESP_OK) break;

    res = httpd_resp_send_chunk(req, "\r\n", 2);
    if (res != ESP_OK) break;
  }

  return res;
}

static esp_err_t flame_handler(httpd_req_t *req) {
  char buf[32];
  snprintf(buf, sizeof(buf), "{\"flame\":%d}", cachedFlame);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_sendstr(req, buf);
}

// ---------- ★ 추가: /discovery 핸들러 ----------
// 서버의 network_scanner()가 POST {"server_ip":"x.x.x.x"} 로 호출
static esp_err_t discovery_handler(httpd_req_t *req) {
  int total = req->content_len;
  if (total <= 0) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":false,\"reason\":\"empty body\"}");
    return ESP_OK;
  }

  // 바디 수신
  std::unique_ptr<char[]> buf(new char[total + 1]);
  int r = httpd_req_recv(req, buf.get(), total);
  if (r <= 0) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  buf[r] = 0;

  // 매우 단순한 파서로 "server_ip":"..." 값 추출 (외부 JSON 라이브러리 의존 제거)
  String body(buf.get());
  int k = body.indexOf("\"server_ip\"");
  if (k >= 0) {
    int colon = body.indexOf(':', k);
    int q1 = body.indexOf('\"', colon + 1);
    int q2 = body.indexOf('\"', q1 + 1);
    if (colon >= 0 && q1 >= 0 && q2 > q1) {
      g_serverIp = body.substring(q1 + 1, q2);
      static bool started = false;
      // 최초 1회만 하트비트 태스크 시작
      if (!started) {
        xTaskCreatePinnedToCore(heartbeat_task, "hb_task", 4096, nullptr, 1, nullptr, 1);
        started = true;
      }
    }
  }

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_sendstr(req, "{\"ok\":true}");
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 8;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  httpd_uri_t jpg_uri = {
    .uri = "/jpg",
    .method = HTTP_GET,
    .handler = jpg_handler,
    .user_ctx = NULL
  };

  httpd_uri_t flame_uri = {
    .uri = "/flame",
    .method = HTTP_GET,
    .handler = flame_handler,
    .user_ctx = NULL
  };

  // ★ 추가: /discovery
  httpd_uri_t discovery_uri = {
    .uri = "/discovery",
    .method = HTTP_POST,
    .handler = discovery_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &flame_uri);
    httpd_register_uri_handler(camera_httpd, &jpg_uri);
    httpd_register_uri_handler(camera_httpd, &discovery_uri); // ★ 등록
  }
}

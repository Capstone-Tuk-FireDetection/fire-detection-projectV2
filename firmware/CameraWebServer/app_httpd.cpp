// app_httpd.cpp (서버 탐색 기능 추가)
#include "esp_http_server.h"
#include "esp_camera.h"
#include "Arduino.h"

extern int cachedFlame;
extern volatile bool allowStreaming;

// [추가] .ino 파일의 onServerIpFound 함수를 호출하기 위한 선언
extern "C" void onServerIpFound(const char* ip);

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// [추가] 서버 탐색 요청을 처리하는 핸들러
static esp_err_t discovery_handler(httpd_req_t *req) {
    char content[100];
    
    int content_len = httpd_req_recv(req, content, sizeof(content) - 1);
    if (content_len <= 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Content is empty");
        return ESP_FAIL;
    }
    content[content_len] = '\0';

    // "server_ip":"<ip>" 형식의 JSON에서 IP 주소 추출
    const char *ip_start = strstr(content, "\"server_ip\":\"");
    if (ip_start) {
        ip_start += strlen("\"server_ip\":\"");
        const char *ip_end = strstr(ip_start, "\"");
        if (ip_end) {
            char received_ip[16];
            int ip_len = ip_end - ip_start;
            if (ip_len < sizeof(received_ip)) {
                strncpy(received_ip, ip_start, ip_len);
                received_ip[ip_len] = '\0';
                
                // .ino 파일의 콜백 함수 호출
                onServerIpFound(received_ip);
                
                httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
                return ESP_OK;
            }
        }
    }

    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
    return ESP_FAIL;
}

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

  // [추가] discovery 핸들러를 위한 URI 구조체
  httpd_uri_t discovery_uri = {
    .uri      = "/discovery",
    .method   = HTTP_POST,
    .handler  = discovery_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &flame_uri);
    httpd_register_uri_handler(camera_httpd, &jpg_uri);
    // [추가] discovery 핸들러 등록
    httpd_register_uri_handler(camera_httpd, &discovery_uri);
  }
}
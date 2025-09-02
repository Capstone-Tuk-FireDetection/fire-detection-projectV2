@ -1,4 +1,4 @@
// app_httpd.cpp (최신 버전, mDNS 대응 포함)
#include "esp_http_server.h"
#include "esp_camera.h"
#include "Arduino.h"
@ -6,6 +6,9 @@
extern int cachedFlame;
extern volatile bool allowStreaming;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
@ -13,6 +16,42 @@ static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

static esp_err_t jpg_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
@ -109,9 +148,19 @@ void startCameraServer() {
    .user_ctx = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &stream_uri);
    httpd_register_uri_handler(camera_httpd, &flame_uri);
    httpd_register_uri_handler(camera_httpd, &jpg_uri);
  }
}
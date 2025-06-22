# esp32-cam
File - preferences 을 클릭하여 Sketchbook location의 경로를 확인하고, 거기에 두 폴더를 삽입해주세요.
기존에 사용하던 라이브러리가 있을경우, library폴더는 제외하고, 따로 받으시면 됩니다.

## WiFi 설정

`CameraWebServer.ino`에서는 `wifi_config.h` 파일을 포함하여 WiFi 정보를 불러옵니다.
해당 파일에서 다음 두 상수를 원하는 값으로 수정하세요.

```C
#define WIFI_SSID      "YOUR_SSID"
#define WIFI_PASSWORD  "YOUR_PASSWORD"
```

스케치를 컴파일하기 전에 `wifi_config.h`가 같은 폴더에 존재해야 합니다.

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

// HTML DOM
// ignore: avoid_web_libraries_in_flutter, deprecated_member_use
import 'dart:html' as html;

// 플랫폼-뷰 레지스트리 (웹 전용 라이브러리)
// ignore: uri_does_not_exist
import 'dart:ui_web' as ui_web;

class WebMjpegView extends StatelessWidget {
  const WebMjpegView({super.key, required this.streamUrl});
  final String streamUrl;

  static const bool enableStreaming = false; // ✅ 스트리밍 비활성화 토글

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb || !enableStreaming) {
      return const Center(child: Text("스트리밍이 비활성화되어 있습니다."));
    }

    final viewType = 'mjpeg-${streamUrl.hashCode}';

    // ignore: undefined_prefixed_name
    ui_web.platformViewRegistry.registerViewFactory(
      viewType,
      (int viewId) {
        final img = html.ImageElement()
          ..src = streamUrl
          ..style.border = 'none'
          ..style.width = '100%'
          ..style.height = '100%';
        return img;
      },
    );

    return HtmlElementView(viewType: viewType);
  }
}

//
// Created by samed on 01.11.18.
//

#include "ScreenshotHandler.h"

ScreenshotHandler::~ScreenshotHandler() {

}

ScreenshotHandler::ScreenshotHandler() {
    renderHeight = 600;
    renderWidth = 800;
    logger = Logger::getInstance();

}

ScreenshotHandler::ScreenshotHandler(int renderHeight, int renderWidth) {
    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;
    logger = Logger::getInstance();

}

bool ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderHeight, renderWidth);
    return true;
}

void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
        const void *buffer, int width, int height) {

    logger->info("Painting!");
}
//
// Created by samed on 01.11.18.
//

#include <include/cef_app.h>
#include "ScreenshotHandler.h"

ScreenshotHandler::~ScreenshotHandler() {

}

ScreenshotHandler::ScreenshotHandler() {
    renderHeight = 600;
    renderWidth = 800;
    logger = Logger::getInstance();

}

ScreenshotHandler::ScreenshotHandler(NodeElement* nodeElement, int renderHeight, int renderWidth) {
    this->nodeElement = nodeElement;
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


    milliseconds invokeTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    invokes.push_back(invokeTime.count());

    logger->info("Painting!");
    logger->info(std::to_string(invokes.front()));
    
    nodeElement = nullptr;
    //CefQuitMessageLoop();
}
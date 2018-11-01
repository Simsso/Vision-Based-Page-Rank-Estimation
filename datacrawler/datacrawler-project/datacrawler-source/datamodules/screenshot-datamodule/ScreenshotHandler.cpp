//
// Created by samed on 01.11.18.
//

#include "ScreenshotHandler.h"

ScreenshotHandler::~ScreenshotHandler() {

}

bool ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, height, width);
    return true;
}
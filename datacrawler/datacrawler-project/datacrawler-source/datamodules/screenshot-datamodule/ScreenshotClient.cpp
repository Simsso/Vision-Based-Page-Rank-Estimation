//
// Created by samed on 01.11.18.
//

#include "ScreenshotClient.h"

ScreenshotClient::ScreenshotClient(){}

ScreenshotClient::~ScreenshotClient(){}

ScreenshotClient::ScreenshotClient(ScreenshotHandler* screenshotHandler){
    this->screenshotHandler = screenshotHandler;
}

CefRefPtr<CefRenderHandler> ScreenshotClient::GetRenderHandler() {
    return screenshotHandler;
}
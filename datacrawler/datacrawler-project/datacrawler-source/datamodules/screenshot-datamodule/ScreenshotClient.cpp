#include "ScreenshotClient.h"

/**
 * ScreenshotClient
 */
ScreenshotClient::ScreenshotClient(){}

/**
 * ~ScreenshotClient
 */
ScreenshotClient::~ScreenshotClient(){}

/**
 * ScreenshotClient - Initializies the client with the custom CefRenderHandler implemented in ScreenshotHandler
 * @param screenshotHandler
 */
ScreenshotClient::ScreenshotClient(ScreenshotHandler* screenshotHandler){
    this->screenshotHandler = screenshotHandler;
}

/**
 * GetRenderHandler - Returns our custom CefRenderHandler written in ScreenshotHandler
 * @return Instance of CefRenderHandler wrapped in a smartpointer CefRefPtr<T>
 */
CefRefPtr<CefRenderHandler> ScreenshotClient::GetRenderHandler() {
    return screenshotHandler;
}
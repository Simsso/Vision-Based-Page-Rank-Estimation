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
ScreenshotClient::ScreenshotClient(ScreenshotHandler* screenshotHandler, ScreenshotRequestHandler* screenshotRequestHandler, ScreenshotLoadhandler * screenshotLoadhandler){
    this->screenshotHandler = screenshotHandler;
    this->screenshotRequestHandler = screenshotRequestHandler;
    this->screenshotLoadhandler = screenshotLoadhandler;
}

/**
 * GetRenderHandler - Returns our custom CefRenderHandler written in ScreenshotHandler
 * @return Instance of CefRenderHandler wrapped in a smartpointer CefRefPtr<T>
 */
CefRefPtr<CefRenderHandler> ScreenshotClient::GetRenderHandler() {
    return screenshotHandler;
}

/**
 * GetRequestHandler - Returns our custom CefRequestHandler written in ScreenshotRequestHandler
 * @return Instance of CefRequestHandler wrapped in a smartpointer CefRefPtr<T>
 */
CefRefPtr<CefRequestHandler> ScreenshotClient::GetRequestHandler() {
    return screenshotRequestHandler;
}

CefRefPtr<CefLoadHandler> ScreenshotClient::GetLoadHandler() { return screenshotLoadhandler;}
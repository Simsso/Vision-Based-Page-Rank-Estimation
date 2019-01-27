#include "ScreenshotHandler.h"

/**
 * ~ScreenshotHandler
 */
ScreenshotHandler::~ScreenshotHandler() {}

/**
 *
 */
ScreenshotHandler::ScreenshotHandler(int renderHeight, int renderWidth) {
    logger = Logger::getInstance();

    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;

    lastScreenshot = new unsigned char[renderHeight * renderWidth * 4];
}

/**
 * GetViewRect - Sets height and width of the given CefRect-instance
 * @param browser represents the current CefBrowser-instance
 * @param rect represents the CefRect-instance of the given CefBrowser-instance
 * @return This will return true.
 */
void ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderWidth, renderHeight);
}

/**
 *
 */
void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
                                const void *buffer, int width, int height) {

    memcpy(lastScreenshot, buffer, sizeof(unsigned char) * height * width * 4);
}

/**
 * getScreenshot - Returns the last screenshot painted.
 *
 * @return Screenshot in size 4 Bytes * width of screenshot * height screenshot
 */
unsigned char* ScreenshotHandler::getScreenshot(){
    return lastScreenshot;
}
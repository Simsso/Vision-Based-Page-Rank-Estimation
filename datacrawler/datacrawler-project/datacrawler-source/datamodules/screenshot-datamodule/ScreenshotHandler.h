#ifndef DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

#include <include/cef_render_handler.h>

#include "../../util/Logger.h"

#include <cmath>
#include <chrono>
#include <mutex>

/**
 * ScreenshotHandler - Implements logic for screenshot taking
 */
class ScreenshotHandler : public CefRenderHandler {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotHandler);
    Logger *logger;

    int renderHeight;
    int renderWidth;
    unsigned char* lastScreenshot;
    bool * quitMessageLoop;


public:
    void GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;

    unsigned char* getScreenshot();

    ScreenshotHandler(int, int, bool*);
    ~ScreenshotHandler() OVERRIDE;
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

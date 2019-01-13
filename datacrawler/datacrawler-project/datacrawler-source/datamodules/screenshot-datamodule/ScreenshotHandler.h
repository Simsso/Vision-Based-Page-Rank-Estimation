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

    bool mHasPainted;
    bool initialInvoke;
    bool * quitMessageLoop;
    std::mutex quitMessageLoopMutex;

    unsigned char* lastScreenshot;

    int32_t numInvokations;
    int64_t sumL1Norm;
    int countLastL1Norms;
    int32_t* lastL1Norms;
    float changePixelThreshold;

    std::chrono::steady_clock::time_point timeOnPaintInvoke;

public:
    void GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;

    unsigned char* calculateChangeMatrix(unsigned char*, unsigned char*, int32_t, int32_t);
    int32_t calculateL1Norm(unsigned char* matrix, int32_t numCol, int32_t numRow);
    void insertL1Norm(int32_t);

    bool hasPainted();
    long getTimeSinceLastPaint();
    unsigned char* getScreenshot();
    std::mutex& getQuitMessageLoopMutex();

    ScreenshotHandler(bool*, int, float, int, int);
    ~ScreenshotHandler() OVERRIDE;
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

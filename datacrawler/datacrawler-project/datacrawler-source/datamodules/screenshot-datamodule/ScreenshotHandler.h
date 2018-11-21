//
// Created by samed on 01.11.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

#include <include/cef_render_handler.h>
#include <include/cef_app.h>

#include "../../util/Logger.h"
#include "../../graph/NodeElement.h"

#include <cmath>
#include <mutex>
#include <list>

class ScreenshotHandler : public CefRenderHandler {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotHandler);
    Logger *logger;
    int renderHeight;
    int renderWidth;
    NodeElement* nodeElement;
    unsigned char* lastScreenshot;
    std::mutex screenshotModuleMutex;
    bool mHasPainted;
    bool initialInvoke;
    int32_t averageL1Distances[4];
    int64_t sumL1Distance;
    int32_t numInvokations;
    vector<double> changeRatesL1distances;


public:
    bool GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;
    int32_t calculateL1Distance(unsigned char*, unsigned char*, int32_t , int32_t);
    double calculateChangeRate();
    void insertAverageL1Distance(int32_t);

    bool hasPainted();
    std::mutex& getMutex();
    
    ScreenshotHandler();
    ScreenshotHandler(NodeElement*, int, int);
    ~ScreenshotHandler();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

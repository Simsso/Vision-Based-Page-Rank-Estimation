//
// Created by samed on 01.11.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

#include <include/cef_render_handler.h>
#include <include/cef_app.h>

#include "../../util/Logger.h"
#include "../../graph/NodeElement.h"

#include <chrono>
#include <list>
#include <cmath>

using namespace std::chrono;

class ScreenshotHandler : public CefRenderHandler {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotHandler);
    Logger *logger;
    int renderHeight;
    int renderWidth;
    NodeElement* nodeElement;
    int32_t deltaNorm;
    int32_t lastL1Norm;

public:
    bool GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;
    int32_t calculateL1Norm(const void*, int32_t , int32_t);
    
    ScreenshotHandler();
    ScreenshotHandler(NodeElement*, int, int);
    ~ScreenshotHandler();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

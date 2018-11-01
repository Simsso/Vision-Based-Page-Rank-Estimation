//
// Created by samed on 01.11.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H


#include <include/cef_render_handler.h>

class ScreenshotHandler : public CefRenderHandler {
private:
    int height;
    int width;

public:
    bool GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;
    
    ScreenshotHandler();
    ScreenshotHandler(int height, int width);
    ~ScreenshotHandler();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTHANDLER_H

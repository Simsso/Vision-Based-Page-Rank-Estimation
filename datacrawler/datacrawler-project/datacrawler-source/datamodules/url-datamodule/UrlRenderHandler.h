//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URLRENDERHANDLER_H
#define DATACRAWLER_PROJECT_URLRENDERHANDLER_H

#include <include/cef_render_handler.h>
#include "../../util/Logger.h"

class UrlRenderHandler: public CefRenderHandler {
private:
    IMPLEMENT_REFCOUNTING(UrlRenderHandler)
    Logger* logger;

public:
    void GetViewRect(CefRefPtr<CefBrowser> , CefRect &) OVERRIDE;
    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType, const RectList &, const void*, int, int) OVERRIDE;

    UrlRenderHandler();
    ~UrlRenderHandler() OVERRIDE;
};

#endif //DATACRAWLER_PROJECT_URLRENDERHANDLER_H

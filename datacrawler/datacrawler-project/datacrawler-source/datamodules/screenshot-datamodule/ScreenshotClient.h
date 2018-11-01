//
// Created by samed on 01.11.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H
#define DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H


#include <include/cef_client.h>
#include "ScreenshotHandler.h"

class ScreenshotClient: public CefClient {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotClient);
    CefRefPtr <CefRenderHandler> screenshotHandler;

public:
    CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE;

    ScreenshotClient();
    ScreenshotClient(ScreenshotHandler*);
    ~ScreenshotClient();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H

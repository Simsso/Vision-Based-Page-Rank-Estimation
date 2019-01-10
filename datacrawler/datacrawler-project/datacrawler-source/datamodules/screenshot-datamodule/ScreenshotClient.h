#ifndef DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H
#define DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H


#include <include/cef_client.h>
#include "ScreenshotHandler.h"
#include "ScreenshotRequestHandler.h"

class ScreenshotClient: public CefClient {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotClient);
    CefRefPtr <CefRenderHandler> screenshotHandler;
    CefRefPtr <CefRequestHandler> screenshotRequestHandler;

public:
    CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE;
    CefRefPtr<CefRequestHandler> GetRequestHandler() OVERRIDE;

    ScreenshotClient();
    ScreenshotClient(ScreenshotHandler*, ScreenshotRequestHandler*);
    ~ScreenshotClient();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H

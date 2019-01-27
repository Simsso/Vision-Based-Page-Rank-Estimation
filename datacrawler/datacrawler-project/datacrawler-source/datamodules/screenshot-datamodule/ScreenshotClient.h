#ifndef DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H
#define DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H

#include <include/cef_client.h>
#include "ScreenshotHandler.h"
#include "ScreenshotRequestHandler.h"
#include "ScreenshotLoadhandler.h"

class ScreenshotClient: public CefClient {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotClient);
    CefRefPtr <CefRenderHandler> screenshotHandler;
    CefRefPtr <CefRequestHandler> screenshotRequestHandler;
    CefRefPtr <CefLoadHandler> screenshotLoadhandler;

public:
    CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE;
    CefRefPtr<CefRequestHandler> GetRequestHandler() OVERRIDE;
    CefRefPtr <CefLoadHandler> GetLoadHandler() OVERRIDE;

    ScreenshotClient();
    ScreenshotClient(ScreenshotHandler*, ScreenshotRequestHandler*, ScreenshotLoadhandler*);

    ~ScreenshotClient();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCLIENT_H

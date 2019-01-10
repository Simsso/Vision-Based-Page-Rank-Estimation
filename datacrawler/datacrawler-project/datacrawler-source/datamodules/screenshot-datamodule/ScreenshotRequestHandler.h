//
// Created by doktorgibson on 1/10/19.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTREQUESTHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTREQUESTHANDLER_H

#include <include/cef_request_handler.h>
#include <iostream>

class ScreenshotRequestHandler : public CefRequestHandler {
IMPLEMENT_REFCOUNTING(ScreenshotRequestHandler);

private:
    std::map<std::string, std::string> headerMap;

public:
    CefRequestHandler::ReturnValue OnBeforeResourceLoad(CefRefPtr<CefBrowser>, CefRefPtr<CefFrame>,
                                                        CefRefPtr<CefRequest>, CefRefPtr<CefRequestCallback>) OVERRIDE;

    ScreenshotRequestHandler(std::map<std::string, std::string>);
    ScreenshotRequestHandler();
    ~ScreenshotRequestHandler() OVERRIDE;

};

#endif //DATACRAWLER_PROJECT_SCREENSHOTREQUESTHANDLER_H

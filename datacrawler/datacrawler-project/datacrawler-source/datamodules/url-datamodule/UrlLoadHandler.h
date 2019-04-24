//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URLLOADHANDLER_H
#define DATACRAWLER_PROJECT_URLLOADHANDLER_H


#include <thread>

#include <include/cef_load_handler.h>
#include "../../util/Logger.h"
#include "UrlDOMVisitor.h"

class UrlLoadHandler : public CefLoadHandler {
private:
    IMPLEMENT_REFCOUNTING(UrlLoadHandler)
    Logger* logger;
    string url;
    bool notified;
    bool failed;
    bool stoppedLoadingStartTime;
    int httpStatusCode;
    std::chrono::steady_clock::time_point loadingStartTime;

public:
    void OnLoadError( CefRefPtr< CefBrowser > browser, CefRefPtr< CefFrame > frame, CefLoadHandler::ErrorCode errorCode, const CefString& errorText, const CefString& failedUrl ) OVERRIDE;
    void OnLoadEnd(CefRefPtr<CefBrowser>, CefRefPtr<CefFrame>, int) OVERRIDE;
    void OnLoadStart(CefRefPtr<CefBrowser> , CefRefPtr<CefFrame> , CefLoadHandler::TransitionType ) OVERRIDE;
    void OnLoadingStateChange(CefRefPtr<CefBrowser>, bool, bool, bool) OVERRIDE;
    UrlLoadHandler(string);
    UrlLoadHandler();
    ~UrlLoadHandler() OVERRIDE;
};


#endif //DATACRAWLER_PROJECT_URLLOADHANDLER_H

//
// Created by doktorgibson on 1/27/19.
//

#ifndef DATACRAWLER_PROJECT_URLREQUESTHANDLER_H
#define DATACRAWLER_PROJECT_URLREQUESTHANDLER_H


#include <include/cef_request_handler.h>
#include "UrlResponseFilter.h"

class UrlRequestHandler : public CefRequestHandler {
        IMPLEMENT_REFCOUNTING(UrlRequestHandler);
private:
    CefRefPtr<CefResponseFilter> urlResponseFilter;
    Logger* logger;

public:
    CefRefPtr<CefResponseFilter> GetResourceResponseFilter( CefRefPtr<CefBrowser>, CefRefPtr<CefFrame>, CefRefPtr<CefRequest>, CefRefPtr<CefResponse>) OVERRIDE;

    UrlRequestHandler(UrlResponseFilter*);
    ~UrlRequestHandler();
};


#endif //DATACRAWLER_PROJECT_URLREQUESTHANDLER_H

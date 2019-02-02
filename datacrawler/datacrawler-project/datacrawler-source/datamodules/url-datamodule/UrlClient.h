//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URLCLIENT_H
#define DATACRAWLER_PROJECT_URLCLIENT_H

#include <include/cef_client.h>
#include <include/cef_app.h>

#include "UrlLoadHandler.h"
#include "UrlRenderHandler.h"
#include "../../util/Logger.h"
#include "UrlCollection.h"
#include "UrlRequestHandler.h"
#include "UrlDisplayHandler.h"

class UrlClient: public CefClient {
private:
    IMPLEMENT_REFCOUNTING(UrlClient)

    Logger* logger;
    UrlCollection* urlCollection;
    CefRefPtr<CefLoadHandler>   urlLoadHandler;
    CefRefPtr<CefRenderHandler>  urlRenderHandler;
    CefRefPtr<CefRequestHandler> urlRequestHandler;
    CefRefPtr<CefDisplayHandler> urlDisplayHandler;
    bool * quitMessageLoop;

public:
    CefRefPtr<CefLoadHandler>    GetLoadHandler()    OVERRIDE;
    CefRefPtr<CefRenderHandler>  GetRenderHandler()  OVERRIDE;
    CefRefPtr<CefRequestHandler> GetRequestHandler() OVERRIDE;
    CefRefPtr<CefDisplayHandler> GetDisplayHandler() OVERRIDE;

    bool OnProcessMessageReceived(CefRefPtr<CefBrowser>, CefProcessId, CefRefPtr<CefProcessMessage>) OVERRIDE;

    UrlClient(string, UrlCollection* , bool* , UrlDisplayHandler*, UrlRequestHandler*, UrlRenderHandler*, UrlLoadHandler*);

    UrlClient();
    ~UrlClient();
};


#endif //DATACRAWLER_PROJECT_URLCLIENT_H

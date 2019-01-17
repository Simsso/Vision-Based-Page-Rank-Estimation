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

class UrlClient: public CefClient {
private:
    IMPLEMENT_REFCOUNTING(UrlClient)

    Logger* logger;
    CefRefPtr <CefLoadHandler> urlLoadHandler;
    CefRefPtr <CefRenderHandler> urlRenderHandler;
    vector<Url*>* urls;

public:
    CefRefPtr<CefLoadHandler> GetLoadHandler() OVERRIDE;
    CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE;
    bool OnProcessMessageReceived(CefRefPtr<CefBrowser>, CefProcessId, CefRefPtr<CefProcessMessage>) OVERRIDE;

    UrlClient(UrlLoadHandler*, UrlRenderHandler*, vector<Url*>*);
    UrlClient();
    ~UrlClient();
};


#endif //DATACRAWLER_PROJECT_URLCLIENT_H

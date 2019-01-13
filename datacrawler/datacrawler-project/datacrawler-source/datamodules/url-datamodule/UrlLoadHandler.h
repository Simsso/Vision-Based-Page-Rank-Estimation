//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URLLOADHANDLER_H
#define DATACRAWLER_PROJECT_URLLOADHANDLER_H

#include <include/cef_load_handler.h>
#include "../../util/Logger.h"
#include "UrlDOMVisitor.h"

class UrlLoadHandler : public CefLoadHandler {
private:
    IMPLEMENT_REFCOUNTING(UrlLoadHandler)
    Logger* logger;
    string url;

public:
    void OnLoadEnd(CefRefPtr<CefBrowser>, CefRefPtr<CefFrame>, int) OVERRIDE;

    UrlLoadHandler(string);
    UrlLoadHandler();
    ~UrlLoadHandler() OVERRIDE;
};


#endif //DATACRAWLER_PROJECT_URLLOADHANDLER_H

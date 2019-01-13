//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_DATACRAWLERRENDERPROCESSHANDLER_H
#define DATACRAWLER_PROJECT_DATACRAWLERRENDERPROCESSHANDLER_H


#include <include/cef_render_process_handler.h>
#include "util/Logger.h"
#include "datamodules/url-datamodule/UrlDOMVisitor.h"

class DatacrawlerRenderProcessHandler: public CefRenderProcessHandler{
private:
    IMPLEMENT_REFCOUNTING(DatacrawlerRenderProcessHandler)
    Logger* logger;
    CefRefPtr<UrlDOMVisitor> urlDOMVisitor;

public:
    bool OnProcessMessageReceived(CefRefPtr<CefBrowser>, CefProcessId, CefRefPtr<CefProcessMessage>) OVERRIDE;

    DatacrawlerRenderProcessHandler();
    ~DatacrawlerRenderProcessHandler();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLERRENDERPROCESSHANDLER_H

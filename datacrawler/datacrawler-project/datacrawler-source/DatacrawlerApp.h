//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_DATACRAWLERAPP_H
#define DATACRAWLER_PROJECT_DATACRAWLERAPP_H


#include <include/cef_app.h>
#include "util/Logger.h"
#include "DatacrawlerRenderProcessHandler.h"

class DatacrawlerApp: public CefApp {
private:
    IMPLEMENT_REFCOUNTING(DatacrawlerApp)
    Logger* logger;
    CefRefPtr<DatacrawlerRenderProcessHandler> datacrawlerRenderProcessHandler;

public:
     CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() OVERRIDE;

     DatacrawlerApp();
     ~DatacrawlerApp();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLERAPP_H

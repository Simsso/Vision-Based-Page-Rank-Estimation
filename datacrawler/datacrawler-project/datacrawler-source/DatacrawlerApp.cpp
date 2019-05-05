//
// Created by doktorgibson on 1/13/19.
//

#include "DatacrawlerApp.h"

DatacrawlerApp::DatacrawlerApp(){
    logger = Logger::getInstance();
    datacrawlerRenderProcessHandler = new DatacrawlerRenderProcessHandler();
}

DatacrawlerApp::~DatacrawlerApp(){}

CefRefPtr<CefRenderProcessHandler> DatacrawlerApp::GetRenderProcessHandler() {
    return datacrawlerRenderProcessHandler;
}

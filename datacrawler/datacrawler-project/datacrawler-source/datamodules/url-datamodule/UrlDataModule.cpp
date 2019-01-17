//
// Created by doktorgibson on 1/12/19.
//

#include "UrlDataModule.h"

UrlDataModule::UrlDataModule() {
    this->numUrls = 10;
    this->urls = new vector<Url*>;
}

UrlDataModule::UrlDataModule(int numUrls) {
    this->numUrls = numUrls;
    this->urls = new vector<Url*>;
}

UrlDataModule::~UrlDataModule() {
    delete urls;
}

DataBase *UrlDataModule::process(std::string url) {
    logger->info("Running URL-DataModule ..");

    logger->info("Requiring UI Thread ..");
    CEF_REQUIRE_UI_THREAD();

    if(!CefCurrentlyOn(TID_UI))
        throw "Not in UI Thread!";

    logger->info("Runnning in UI thread!");

    bool * quitMessageLoop = new bool;
    *quitMessageLoop = false;

    CefRefPtr<UrlRenderHandler> urlRenderHandler(new UrlRenderHandler());
    CefRefPtr<UrlLoadHandler> urlLoadHandler(new UrlLoadHandler(url, numUrls));

    CefRefPtr<UrlClient> urlClient(new UrlClient(urlLoadHandler, urlRenderHandler, urls));

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 60;

    CefBrowserHost::CreateBrowser(cefWindowInfo, urlClient.get(), url, browserSettings, NULL);

    CefRunMessageLoop();

    for(auto x: *urls){
        logger->info("parsed: "+x->getUrl());
    }
    return nullptr;
}

//
// Created by doktorgibson on 1/12/19.
//

#include "UrlDataModule.h"

UrlDataModule::UrlDataModule() {
    this->numUrls = 10;
}

UrlDataModule::UrlDataModule(int numUrls) {
    this->numUrls = numUrls;
}

UrlDataModule::~UrlDataModule() {
}

DataBase *UrlDataModule::process(std::string url) {
    logger->info("Running URL-DataModule ..");

    logger->info("Requiring UI Thread ..");
    CEF_REQUIRE_UI_THREAD();

    if(!CefCurrentlyOn(TID_UI))
        throw "Not in UI Thread!";

    logger->info("Runnning in UI thread!");

    UrlCollection * urlCollection = new UrlCollection();
    size_t * totalSize = new size_t;
    *totalSize = 0;

    CefRefPtr<UrlResponseFilter> urlResponseFilter(new UrlResponseFilter(totalSize));
    CefRefPtr<UrlRequestHandler> urlRequestHandler(new UrlRequestHandler(urlResponseFilter));
    CefRefPtr<UrlRenderHandler> urlRenderHandler(new UrlRenderHandler());
    CefRefPtr<UrlLoadHandler> urlLoadHandler(new UrlLoadHandler(url, numUrls));
    CefRefPtr<UrlClient> urlClient(new UrlClient(urlLoadHandler, urlRenderHandler, urlCollection, urlRequestHandler));

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 300;

    CefRefPtr<CefBrowser> browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, urlClient.get(), url, browserSettings, NULL);

    CefRunMessageLoop();

    urlCollection->setSize(*totalSize / 1024);

    browser->GetHost()->CloseBrowser(true);
    delete totalSize;
    return urlCollection;
}

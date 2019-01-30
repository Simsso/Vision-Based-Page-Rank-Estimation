//
// Created by doktorgibson on 1/12/19.
//

#include "UrlDataModule.h"

UrlDataModule::UrlDataModule() {
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

    UrlCollection * urlCollection = new UrlCollection;
    size_t totalSize = 0;
    string title = "";

    CefRefPtr<UrlClient> urlClient(new UrlClient(url, urlCollection, title, totalSize));

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 1;

    CefRefPtr<CefBrowser> browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, urlClient.get(), url, browserSettings, NULL);

    CefRunMessageLoop();

    urlCollection->setSize(totalSize / 1024);
    urlCollection->setTitle(title);
    browser->GetHost()->CloseBrowser(true);

    return urlCollection;
}

//
// Created by samed on 23.10.18.
//

#include "datacrawler.h"

void Datacrawler::init() {
        logger->info("Initialising Datacrawler !");

        if(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE) != nullptr){
            dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE)->createInstance());
            logger->info("Using ScreenshotDataModule ..");
        }

        if(datacrawlerConfiguration.getConfiguration(URL_MODULE) != nullptr){
            dataModules.push_front(datacrawlerConfiguration.getConfiguration(URL_MODULE)->createInstance());
            logger->info("Using URLDataModule ..");
        }

        logger->info("Initialising Datacrawler finished!");
}

bool Datacrawler::process() {
    logger->info("Running DataModules!");

    for(auto  x: dataModules){
        x->process();
    }
    /*CefMainArgs mainArgs(argc, argv);
    CefExecuteProcess(mainArgs, NULL, NULL);

    CefSettings cefSettings;

    // Initialize CEF for the browser process
    CefInitialize(mainArgs, cefSettings, NULL, NULL);

    ExperimentHandler* experimentHandler = new ExperimentHandler(1920, 1080);

    // CefRefPtr represents a SmartPointer (Releases Object once function returns)
    //
    // ExperimentClient implements application-level callbacks for the browser process
    // pass the custom CefRenderHandler object
    CefRefPtr<ExperimentClient> client = new ExperimentClient(experimentHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;

    // Create a synchronous browser, will be created once CEF has initialized
    CefRefPtr<CefBrowser> browser;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, client.get(), "https://timodenk.com", browserSettings, NULL);
    browser->
            CefRunMessageLoop();

    CefShutdown(); */

    return true;
}
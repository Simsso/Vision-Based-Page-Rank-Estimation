//
// Created by Samed Güner on 18.10.18.
//

#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>
#include "datacrawler.h"

Logger* Logger::instance = 0;

int main(int argc, char* argv[]){

    CefMainArgs mainArgs(argc, argv);
    CefSettings cefSettings;

    CefExecuteProcess(mainArgs, NULL, NULL);

    CefInitialize(mainArgs, cefSettings, NULL, NULL);

    Logger* logger  = Logger::getInstance();
    logger->info("Initialization of CEF finished!");

    logger->info("Requiring UI Thread for Datacrawler ..");
    CEF_REQUIRE_UI_THREAD();

    if(CefCurrentlyOn(TID_UI)) {
        logger->info("Runnning in UI thread!");
        logger->info("Starting Datacrawler !");

        Datacrawler datacrawler;
        datacrawler.init();

        datacrawler.process("https://google.com/");
        datacrawler.process("https://google.com/");
        logger->info("Datacrawler execution finished!");
    }

    logger->info("Shutting down CEF!");
    CefShutdown();
}

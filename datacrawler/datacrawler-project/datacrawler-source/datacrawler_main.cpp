#include "datacrawler.h"

Logger *Logger::instance = 0;

#include "stdlib.h"
#include "GraphOutput.h"
#include "DatacrawlerApp.h"

int main(int argc, char *argv[]) {
    setenv("LOG_LEVEL", "LOG_ALL", true);
    Logger *logger = Logger::getInstance();

    CefMainArgs mainArgs(argc, argv);

    CefRefPtr<DatacrawlerApp> datacrawlerApp(new DatacrawlerApp());

    int exitCode = CefExecuteProcess(mainArgs, datacrawlerApp, NULL);
    if (exitCode >= 0) {
        // The sub-process terminated, exit now.
        return exitCode;
    }

    CefSettings cefSettings;
    cefSettings.windowless_rendering_enabled = true;

    if(CefInitialize(mainArgs, cefSettings, NULL, NULL))
        logger->info("Initializing CEF finished .. !");
    else {
        logger->fatal("Initializing has failed!");
    }

    Datacrawler datacrawler;
    datacrawler.init();

    string domains[] { "https://www.google.com/calendar?tab=wc" , "facebook.com", "samedguener.com", "timodenk.com", "dhbw.de", "sap.com" };

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    for(auto x: domains){
        map<string, NodeElement*> * graph;
        GraphOutput* graphOutput;

        graph = datacrawler.process(x);
        graphOutput = new GraphOutput(graph, "1");
        graphOutput->generateGraph();

        delete graph;
        delete graphOutput;
    }

    long delta = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    logger->info("It took:"+ std::to_string(delta/6));
    logger->info("Datacrawler execution finished!");
    CefShutdown();
}
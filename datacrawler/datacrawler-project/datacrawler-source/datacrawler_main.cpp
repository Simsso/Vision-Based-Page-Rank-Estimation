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
    map<string, NodeElement*> * graph;
    GraphOutput* graphOutput;

    graph = datacrawler.process("youtube.com");
    graphOutput = new GraphOutput(graph, "2");
    graphOutput->generateGraph();

    delete graph;
    delete graphOutput;

    logger->info("Datacrawler execution finished!");
    CefShutdown();
}
#include "datacrawler.h"

#define DATASET_VERSION_1

Logger *Logger::instance = 0;

#ifdef DATASET_VERSION_1

int main(int argc, char *argv[]) {

    CefMainArgs mainArgs(argc, argv);
    CefExecuteProcess(mainArgs, NULL, NULL);

    Logger *logger = Logger::getInstance();

    logger->info("Starting Datacrawler !");

    char *url = std::getenv("URL");

    Datacrawler datacrawler(&mainArgs);
    datacrawler.init();

    datacrawler.process(url);
    logger->info("Datacrawler execution finished!");
}

#endif // DATASET_VERSION_1

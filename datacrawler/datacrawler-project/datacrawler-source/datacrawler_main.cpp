#include "datacrawler.h"

#define DATASET_VERSION_1

Logger *Logger::instance = 0;

#ifdef DATASET_VERSION_1

int main(int argc, char *argv[]) {
    Logger *logger = Logger::getInstance();
    char *url = std::getenv("URL");
    char *urlRank = getenv("URL_RANK");
    char *outputPath = getenv("OUTPUT_PATH");

    if(urlRank == NULL){
     logger->fatal("URL_RANK not specified! Exiting!");
     exit(1);
    }

    if(outputPath == NULL){
     logger->fatal("OUTPUT_PATH not specified! Exiting!");
     exit(1);
    }

    long rank = atol(urlRank);

    CefMainArgs mainArgs(argc, argv);
    CefExecuteProcess(mainArgs, NULL, NULL);

    logger->info("Starting Datacrawler !");

    Datacrawler datacrawler(&mainArgs);

    datacrawler.init();

    NodeElement * node = datacrawler.process(url);

    vector<DataBase*>* dataBase = node->getData();

    for(auto x: *dataBase){
        DataModulesEnum dataModulesEnum = x->getDataModuleType();

        if(dataModulesEnum == SCREENSHOT_MODULE || dataModulesEnum == SCREENSHOT_MOBILE_MODULE) {
            logger->info("We have screenshots! Saving to disk!");
        } else {
            logger->info("No data has been produced!");
        }
    }

    logger->info("Datacrawler execution finished!");
}

#endif // DATASET_VERSION_1

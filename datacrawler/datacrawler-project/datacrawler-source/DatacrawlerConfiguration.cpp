#include "DatacrawlerConfiguration.h"

/**
 * ~DataCrawlerConfiguration
 */
DatacrawlerConfiguration::~DatacrawlerConfiguration() {
    for (auto datacrawlerConfiguration: configurations) {
        delete datacrawlerConfiguration.second;
    }
}

/**
 * DatacrawlerConfiguration - Loads all user-defined configurations for the graph to be generated
 */
DatacrawlerConfiguration::DatacrawlerConfiguration() {
    logger = Logger::getInstance();

    // TODO Refactor, add default values if not available
    char *datamoduleName = std::getenv("DATAMODULE");

    if(datamoduleName == NULL) {
        logger->fatal("No datamodule has been detected! Please specify the environment variable <DATAMODULE>!");
        logger->fatal("Exiting datacrawler!");
        exit(1);
    } else if (toStringDataModulesEnum(SCREENSHOT_MODULE).compare(datamoduleName) == 0) {
        logger->info("SCREENSHOT_MODULE has been activated!");

        char * ONPAINT_TIMEOUT_ENV = std::getenv("ONPAINT_TIMEOUT");
        if(ONPAINT_TIMEOUT_ENV == NULL){
            logger->fatal("ONPAINT_TIMEOUT not specified! Exiting!");
            exit(1);
        }
        long ONPAINT_TIMEOUT = atol(ONPAINT_TIMEOUT_ENV);

        char * ELAPSED_TIME_ONPAINT_TIMEOUT_ENV = std::getenv("ELAPSED_TIME_ONPAINT_TIMEOUT");
        if(ELAPSED_TIME_ONPAINT_TIMEOUT_ENV == NULL){
            logger->fatal("ELAPSED_TIME_ONPAINT_TIMEOUT not specified! Exiting!");
            exit(1);
        }
        long ELAPSED_TIME_ONPAINT_TIMEOUT = atol(ELAPSED_TIME_ONPAINT_TIMEOUT_ENV);

        char * LAST_SCREENSHOTS_ENV = std::getenv("LAST_SCREENSHOTS");
        if(LAST_SCREENSHOTS_ENV == NULL){
            logger->fatal("LAST_SCREENSHOTS not specified! Exiting!");
            exit(1);
        }
        long LAST_SCREENSHOTS = atol(LAST_SCREENSHOTS_ENV);

        char * CHANGE_THRESHOLD_ENV = std::getenv("CHANGE_THRESHOLD");
        if(CHANGE_THRESHOLD_ENV == NULL){
            logger->fatal("CHANGE_THRESHOLD not specified! Exiting!");
            exit(1);
        }
        double CHANGE_THRESHOLD = atof(CHANGE_THRESHOLD_ENV);

        configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(1080, 1920, ONPAINT_TIMEOUT, ELAPSED_TIME_ONPAINT_TIMEOUT, CHANGE_THRESHOLD, LAST_SCREENSHOTS, false);
    } else if (toStringDataModulesEnum(SCREENSHOT_MOBILE_MODULE).compare(datamoduleName) == 0) {
        logger->info("SCREENSHOT_MOBILE_MODULE has been activated!");

        char * ONPAINT_TIMEOUT_ENV = std::getenv("ONPAINT_TIMEOUT");
        if(ONPAINT_TIMEOUT_ENV == NULL){
            logger->fatal("ONPAINT_TIMEOUT not specified! Exiting!");
            exit(1);
        }
        long ONPAINT_TIMEOUT = atol(ONPAINT_TIMEOUT_ENV);

        char * ELAPSED_TIME_ONPAINT_TIMEOUT_ENV = std::getenv("ELAPSED_TIME_ONPAINT_TIMEOUT");
        if(ELAPSED_TIME_ONPAINT_TIMEOUT_ENV == NULL){
            logger->fatal("ELAPSED_TIME_ONPAINT_TIMEOUT not specified! Exiting!");
            exit(1);
        }
        long ELAPSED_TIME_ONPAINT_TIMEOUT = atol(ELAPSED_TIME_ONPAINT_TIMEOUT_ENV);

        char * LAST_SCREENSHOTS_ENV = std::getenv("LAST_SCREENSHOTS");
        if(LAST_SCREENSHOTS_ENV == NULL){
            logger->fatal("LAST_SCREENSHOTS not specified! Exiting!");
            exit(1);
        }
        long LAST_SCREENSHOTS = atol(LAST_SCREENSHOTS_ENV);

        char * CHANGE_THRESHOLD_ENV = std::getenv("CHANGE_THRESHOLD");
        if(CHANGE_THRESHOLD_ENV == NULL){
            logger->fatal("CHANGE_THRESHOLD not specified! Exiting!");
            exit(1);
        }
        double CHANGE_THRESHOLD = atof(CHANGE_THRESHOLD_ENV);

        configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(400, 400, ONPAINT_TIMEOUT, ELAPSED_TIME_ONPAINT_TIMEOUT, CHANGE_THRESHOLD, LAST_SCREENSHOTS, true);
    }
}

/**
 * getConfiguration
 * @param selectedConfig represents the name of the DataModule for whom configuration shall be loaded
 * @return DataModuleBaseConfiguration, which represents the base class of all configurations for DataModules
 */
DataModuleBaseConfiguration *DatacrawlerConfiguration::getConfiguration(DataModulesEnum selectedConfig) {
    try {
        return configurations.at(selectedConfig);
    } catch (std::out_of_range ex) {
        return nullptr;
    }
}
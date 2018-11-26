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

    // TODO Figure out how to pass complex datasctructures (derived objects) through IPC to allow multiple datamodules in one process
    char *datamoduleName = std::getenv("DATAMODULE");

    if(datamoduleName == NULL) {
        logger->fatal("No datamodule has been detected! Please specify the environment variable <DATAMODULE>!");
        logger->fatal("Exiting datacrawler!");
        exit(1);
    } else if (toStringDataModulesEnum(SCREENSHOT_MODULE).compare(datamoduleName) == 0) {
        logger->info("SCREENSHOT_MODULE has been activated!");
        configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(1920, 1080, false);
    } else if (toStringDataModulesEnum(SCREENSHOT_MOBILE_MODULE).compare(datamoduleName) == 0) {
        logger->info("SCREENSHOT_MOBILE_MODULE has been activated!");
        configurations[SCREENSHOT_MOBILE_MODULE] = new ScreenshotConfiguration(400, 400, true);
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
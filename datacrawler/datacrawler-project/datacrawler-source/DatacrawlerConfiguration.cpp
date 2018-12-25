#include "DatacrawlerConfiguration.h"

/**
 * ~DataCrawlerConfiguration
 */
DatacrawlerConfiguration::~DatacrawlerConfiguration() {
    for(auto datacrawlerConfiguration: configurations){
        delete datacrawlerConfiguration.second;
    }
}

/**
 * DatacrawlerConfiguration - Loads all user-defined configurations for the graph to be generated
 */
DatacrawlerConfiguration::DatacrawlerConfiguration() {
    configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(1920, 1080, false);
    //configurations[SCREENSHOT_MOBILE_MODULE] = new ScreenshotConfiguration(400, 400, true);
}

/**
 * getConfiguration
 * @param selectedConfig represents the name of the DataModule for whom configuration shall be loaded
 * @return DataModuleBaseConfiguration, which represents the base class of all configurations for DataModules
 */
DataModuleBaseConfiguration* DatacrawlerConfiguration::getConfiguration(DataModulesEnum selectedConfig) {
    try {
        return configurations.at(selectedConfig);
    } catch(std::out_of_range ex){
        return nullptr;
    }
}
//
// Created by samed on 23.10.18.
//

#include "DatacrawlerConfiguration.h"

// TODO Clean-up heap
DatacrawlerConfiguration::~DatacrawlerConfiguration() {}

DatacrawlerConfiguration::DatacrawlerConfiguration() {
    configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(800,600);
}

DataModuleBaseConfiguration* DatacrawlerConfiguration::getConfiguration(DataModulesEnum selectedConfig) {
    try {
        return configurations.at(selectedConfig);
    } catch(std::out_of_range ex){
        return nullptr;
    }
}
#ifndef DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H
#define DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

#include "map"
#include "DataModulesEnum.h"
#include "datamodule-configs/DataModuleBaseConfiguration.h"
#include "datamodule-configs/ScreenshotConfiguration.h"

using namespace std;

class DatacrawlerConfiguration {

private:
    map<DataModulesEnum, DataModuleBaseConfiguration*> configurations;

public:
    DataModuleBaseConfiguration* getConfiguration(DataModulesEnum);

    DatacrawlerConfiguration();
    ~DatacrawlerConfiguration();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

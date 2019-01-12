#ifndef DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H
#define DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

#include "map"
#include "DataModulesEnum.h"
#include "datamodule-configs/DataModuleBaseConfiguration.h"
#include "datamodule-configs/ScreenshotConfiguration.h"
#include "util/json.hpp"
#include "fstream"

using namespace std;
using namespace nlohmann;

class DatacrawlerConfiguration {

private:
    map<DataModulesEnum, DataModuleBaseConfiguration*> configurations;
    Logger* logger;

    ScreenshotConfiguration* generateScreenshotDatamoduleConfig(json&);
    ScreenshotConfiguration* genereateScreenshotMobileDatamoduleConfig(json&);
public:
    DataModuleBaseConfiguration* getConfiguration(DataModulesEnum);

    DatacrawlerConfiguration();
    ~DatacrawlerConfiguration();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

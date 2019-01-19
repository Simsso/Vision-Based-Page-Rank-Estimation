#ifndef DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H
#define DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

#include "DataModulesEnum.h"
#include "datamodule-configs/DataModuleBaseConfiguration.h"
#include "datamodule-configs/ScreenshotConfiguration.h"
#include "datamodule-configs/UrlConfiguration.h"

#include "map"
#include "util/json.hpp"
#include "fstream"

using namespace std;
using namespace nlohmann;

class DatacrawlerConfiguration {

private:
    map<DataModulesEnum, DataModuleBaseConfiguration*> configurations;
    Logger* logger;
    int numNodes;

    ScreenshotConfiguration* generateScreenshotDatamoduleConfig(json&, bool);
    UrlConfiguration* generateUrlDataModuleConfig(json&);

public:
    DataModuleBaseConfiguration* getConfiguration(DataModulesEnum);
    int getNumNodes();
    DatacrawlerConfiguration();
    ~DatacrawlerConfiguration();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLERCONFIGURATION_H

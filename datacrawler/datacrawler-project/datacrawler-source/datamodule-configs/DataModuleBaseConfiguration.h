#ifndef DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H
#define DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H


#include "../datamodules/DataModuleBase.h"

class DataModuleBaseConfiguration {

public:
     virtual DataModuleBase* createInstance();

     DataModuleBaseConfiguration();
     virtual ~DataModuleBaseConfiguration();
};


#endif //DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H

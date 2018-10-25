//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H
#define DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H


#include "../datamodules/DataModuleBase.h"

class DataModuleBaseConfiguration {

public:
     virtual DataModuleBase* createInstance();

     DataModuleBaseConfiguration();
     ~DataModuleBaseConfiguration();
};


#endif //DATACRAWLER_PROJECT_DATAMODULECONFIGURATION_H

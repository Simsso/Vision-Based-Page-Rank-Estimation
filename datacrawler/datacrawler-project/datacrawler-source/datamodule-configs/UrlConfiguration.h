//
// Created by doktorgibson on 1/12/19.
//

#ifndef DATACRAWLER_PROJECT_URLCONFIGURATION_H
#define DATACRAWLER_PROJECT_URLCONFIGURATION_H

#include "DataModuleBaseConfiguration.h"
#include "../datamodules/url-datamodule/UrlDataModule.h"

class UrlConfiguration: public DataModuleBaseConfiguration {

public:
    DataModuleBase* createInstance();

    UrlConfiguration();
    ~UrlConfiguration();
};


#endif //DATACRAWLER_PROJECT_URLCONFIGURATION_H

//
// Created by doktorgibson on 1/12/19.
//

#ifndef DATACRAWLER_PROJECT_URLCONFIGURATION_H
#define DATACRAWLER_PROJECT_URLCONFIGURATION_H

#include "DataModuleBaseConfiguration.h"
#include "../datamodules/url-datamodule/UrlDataModule.h"

class UrlConfiguration: public DataModuleBaseConfiguration {
private:
    int numUrls;

public:
    DataModuleBase* createInstance();

    UrlConfiguration();
    UrlConfiguration(int);
    ~UrlConfiguration();
};


#endif //DATACRAWLER_PROJECT_URLCONFIGURATION_H

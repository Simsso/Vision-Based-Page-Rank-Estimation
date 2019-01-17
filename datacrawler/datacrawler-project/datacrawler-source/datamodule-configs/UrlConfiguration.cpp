//
// Created by doktorgibson on 1/12/19.
//

#include "UrlConfiguration.h"

UrlConfiguration::UrlConfiguration(){};

UrlConfiguration::UrlConfiguration(int numUrls){
    this->numUrls = numUrls;
}

UrlConfiguration::~UrlConfiguration(){};

DataModuleBase* UrlConfiguration::createInstance() {
    return new UrlDataModule(numUrls);
}

//
// Created by doktorgibson on 1/12/19.
//

#include "UrlConfiguration.h"

UrlConfiguration::UrlConfiguration(){};
UrlConfiguration::~UrlConfiguration(){};

DataModuleBase* UrlConfiguration::createInstance() {
    return new UrlDataModule();
}

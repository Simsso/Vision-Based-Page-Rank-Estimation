//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_DATACRAWLER_H
#define DATACRAWLER_PROJECT_DATACRAWLER_H

#include <iostream>
#include <list>
#include <include/internal/cef_linux.h>
#include "DatacrawlerConfiguration.h"
#include "datamodules/DataModuleBase.h"
#include "datamodules/ScreenshotDataModule.h"
#include "util/Logger.h"

using namespace std;

class Datacrawler {

private:
   string URL;
   DatacrawlerConfiguration datacrawlerConfiguration;
   list<DataModuleBase*> dataModules;
   Logger* logger;

public:
    Datacrawler(){
        logger = Logger::getInstance();
    }

    Datacrawler(string URL){
        this->URL = URL;
        logger = Logger::getInstance();
    }

    bool process();
    void init();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLER_H

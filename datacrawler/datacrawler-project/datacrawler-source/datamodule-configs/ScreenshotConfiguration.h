//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H
#define DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

#include "DataModuleBaseConfiguration.h"
#include "../datamodules/ScreenshotDataModule.h"

class ScreenshotConfiguration: public DataModuleBaseConfiguration {
private:
    int height;
    int width;

public:
    DataModuleBase* createInstance();

    ScreenshotConfiguration();
    ScreenshotConfiguration(int width, int height);
    ~ScreenshotConfiguration();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

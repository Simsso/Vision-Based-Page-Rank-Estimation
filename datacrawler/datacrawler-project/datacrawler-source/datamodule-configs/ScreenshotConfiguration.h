#ifndef DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H
#define DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

#include "DataModuleBaseConfiguration.h"
#include "../datamodules/screenshot-datamodule/ScreenshotDataModule.h"

class ScreenshotConfiguration: public DataModuleBaseConfiguration {
private:
    int height;
    int width;
    bool mobile;

public:
    DataModuleBase* createInstance();

    ScreenshotConfiguration();
    ScreenshotConfiguration(int, int, bool);
    ~ScreenshotConfiguration();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

#ifndef DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H
#define DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

#include "DataModuleBaseConfiguration.h"
#include "../datamodules/screenshot-datamodule/ScreenshotDataModule.h"

class ScreenshotConfiguration: public DataModuleBaseConfiguration {
private:
    int height;
    int width;
    bool mobile;
    int onPaintTimeout;
    int elapsedTimeOnPaintTimeout;
    int lastScreenshots;
    double changeThreshold;

public:
    DataModuleBase* createInstance();

    ScreenshotConfiguration();
    ScreenshotConfiguration(int, int, int, int, double, int, bool);
    ~ScreenshotConfiguration();

};


#endif //DATACRAWLER_PROJECT_SCREENSHOTCONFIGURATION_H

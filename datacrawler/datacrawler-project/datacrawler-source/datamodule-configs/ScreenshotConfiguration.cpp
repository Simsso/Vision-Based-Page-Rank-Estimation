#include "ScreenshotConfiguration.h"

/**
 * ~ScreenshotConfiguration
 */
ScreenshotConfiguration::~ScreenshotConfiguration(){}

/**
 * ScreenshotConfiguration - Standard configuration of the ScreenshotDataModule with 800x600 as screenshot size
 */
ScreenshotConfiguration::ScreenshotConfiguration(){
    width = 1920;
    height = 1080;
    mobile = false;
}

// TODO refactor parameter
/**
 * ScreenshotConfiguration - Constructor to configure the ScreenshotDataModule
 * @param width specifies the width of the screenshot to be taken
 * @param height specifies the height of the screenshot to be taken
 */
ScreenshotConfiguration::ScreenshotConfiguration(int height, int width, int onPaintTimeout, int elapsedTimeOnPaintTimeout, double changeThreshold, int lastScreenshots, bool mobile) {
    this->width  = width;
    this->height = height;
    this->mobile = mobile;
    this->onPaintTimeout = onPaintTimeout;
    this->lastScreenshots = lastScreenshots;
    this->changeThreshold = changeThreshold;
    this->elapsedTimeOnPaintTimeout = elapsedTimeOnPaintTimeout;
}

/**
 * createInstance
 * @return instance of the ScreenshotDataModule
 */
DataModuleBase* ScreenshotConfiguration::createInstance() {
    return new ScreenshotDataModule(height, width, onPaintTimeout, elapsedTimeOnPaintTimeout, changeThreshold, lastScreenshots, mobile);
}


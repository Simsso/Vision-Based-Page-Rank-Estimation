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

/**
 * ScreenshotConfiguration - Constructor to configure the ScreenshotDataModule
 * @param width specifies the width of the screenshot to be taken
 * @param height specifies the height of the screenshot to be taken
 */
ScreenshotConfiguration::ScreenshotConfiguration(int width, int height, bool mobile){
    this->width  = width;
    this->height = height;
    this->mobile = mobile;
}

/**
 * createInstance
 * @return instance of the ScreenshotDataModule
 */
DataModuleBase* ScreenshotConfiguration::createInstance() {
    return new ScreenshotDataModule(height, width, mobile);
}


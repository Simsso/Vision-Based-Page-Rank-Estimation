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
ScreenshotConfiguration::ScreenshotConfiguration(int height, int width, int ONPAINT_TIMEOUT, int ELAPSED_TIME_ONPAINT_TIMEOUT, double CHANGE_THRESHOLD, int LAST_SCREENSHOTS, bool mobile) {
    this->width  = width;
    this->height = height;
    this->mobile = mobile;
    this->ONPAINT_TIMEOUT = ONPAINT_TIMEOUT;
    this->LAST_SCREENSHOTS = LAST_SCREENSHOTS;
    this->CHANGE_THRESHOLD = CHANGE_THRESHOLD;
    this->ELAPSED_TIME_ONPAINT_TIMEOUT = ELAPSED_TIME_ONPAINT_TIMEOUT;
}

/**
 * createInstance
 * @return instance of the ScreenshotDataModule
 */
DataModuleBase* ScreenshotConfiguration::createInstance() {
    return new ScreenshotDataModule(height, width, ONPAINT_TIMEOUT, ELAPSED_TIME_ONPAINT_TIMEOUT, CHANGE_THRESHOLD, LAST_SCREENSHOTS, mobile);
}


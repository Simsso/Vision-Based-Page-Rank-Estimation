#include "ScreenshotData.h"

/**
 *  ~ScreenshotData
 */
ScreenshotData::~ScreenshotData(){
    for(Screenshot* screenshot: screenshots){
        delete screenshot;
    }
}

/**
 * getScreenshots
 * @return return all Screenshot instances
 */
std::vector<Screenshot*>  ScreenshotData::getScreenshots() { return screenshots;}

/**
 * addScreenshot - Creates a Screenshot-instance and add it to the internal screenshots structure
 * @param screenshot, which shall be added
 * @param height of the screenshot
 * @param width of the screenshot
 */
void ScreenshotData::addScreenshot(unsigned char *screenshot, int height, int width) {
   addScreenshot(new Screenshot(screenshot, height, width));
}

/**
 * addScreenshot - Adds an existing Screenshot-instance to the internal screenshots structure
 * @param screenshot
 */
void ScreenshotData::addScreenshot(Screenshot* screenshot) {
    screenshots.push_back(screenshot);
}

/**
 * getDataModuleType
 * @return Value of DataModulesEnum, which shows from which data this DataBase derivate actually is
 */
DataModulesEnum ScreenshotData::getDataModuleType() { return SCREENSHOT_MODULE;}
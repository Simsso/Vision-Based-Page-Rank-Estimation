#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATA_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATA_H

#include "Screenshot.h"
#include "../DataBase.h"
#include <vector>

class ScreenshotData: public DataBase {
private:
    std::vector<Screenshot*> screenshots;
    bool mobile;

public:
    std::vector<Screenshot*> getScreenshots();
    void addScreenshot(unsigned char*, int, int);
    void addScreenshot(Screenshot*);

    DataModulesEnum getDataModuleType();

    ScreenshotData(bool);
    ~ScreenshotData();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATA_H

//
// Created by doktorgibson on 11/22/18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATA_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATA_H

#include "Screenshot.h"
#include "../DataBase.h"
#include <vector>

class ScreenshotData: public DataBase {
private:
    std::vector<Screenshot*> screenshots;

public:
    std::vector<Screenshot*> getScreenshots();
    void addScreenshot(unsigned char*, int, int);
    void addScreenshot(Screenshot*);

    DataModulesEnum getDataModuleType();

    ~ScreenshotData();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATA_H

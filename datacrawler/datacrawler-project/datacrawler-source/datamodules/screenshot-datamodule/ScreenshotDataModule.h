//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include "../DataModuleBase.h"
#include "../../datamodule-configs/ScreenshotConfiguration.h"

class ScreenshotDataModule : public DataModuleBase {
private:
    int height;
    int width;

public:
    NodeElement* process(string);

    ScreenshotDataModule();
    ScreenshotDataModule(int, int);
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

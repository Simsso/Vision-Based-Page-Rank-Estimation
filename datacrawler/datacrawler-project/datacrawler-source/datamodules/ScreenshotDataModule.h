//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include "DataModuleBase.h"

class ScreenshotDataModule : public DataModuleBase {

public:
    NodeElement* process();

    ScreenshotDataModule();
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

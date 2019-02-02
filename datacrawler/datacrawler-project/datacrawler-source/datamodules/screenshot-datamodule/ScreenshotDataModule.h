//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include <thread>
#include <chrono>

#include <include/internal/cef_ptr.h>
#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>

#include "../DataModuleBase.h"
#include "ScreenshotHandler.h"
#include "ScreenshotRequestHandler.h"
#include "ScreenshotClient.h"
#include "Screenshot.h"

class ScreenshotDataModule : public DataModuleBase, public CefBaseRefCounted {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotDataModule);
    int height;
    int width;
    bool mobile;
    int onPaintTimeout;
    int elapsedTimeOnPaintTimeout;

public:
    DataBase * process(std::string) OVERRIDE;

    ScreenshotDataModule();
    ScreenshotDataModule(int, int, int, int, bool);
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

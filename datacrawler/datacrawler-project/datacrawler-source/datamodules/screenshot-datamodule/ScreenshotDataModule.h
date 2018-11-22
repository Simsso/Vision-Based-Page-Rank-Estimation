//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include <thread>
#include <chrono>
#include <mutex>


#include <include/internal/cef_ptr.h>
#include "../DataModuleBase.h"
#include "ScreenshotHandler.h"
#include "ScreenshotClient.h"
#include "Screenshot.h"
#include "ScreenshotData.h"

#define ONPAINT_TIMEOUT 10
#define ELAPSED_TIME_ONPAINT_TIMEOUT 2500

class ScreenshotDataModule : public DataModuleBase {
private:
    int height;
    int width;
    ScreenshotHandler* screenshotHandler;
    CefRefPtr<ScreenshotClient> screenshotClient;
    CefRefPtr<CefBrowser> browser;

public:
    DataBase* process(std::string);

    ScreenshotDataModule();
    ScreenshotDataModule(int, int);
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

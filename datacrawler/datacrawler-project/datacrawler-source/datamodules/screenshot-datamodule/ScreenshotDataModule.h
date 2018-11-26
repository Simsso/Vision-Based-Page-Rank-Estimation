//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include <thread>
#include <chrono>
#include <mutex>


#include <include/internal/cef_ptr.h>
#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>

#include "../DataModuleBase.h"
#include "ScreenshotHandler.h"
#include "ScreenshotClient.h"
#include "Screenshot.h"
#include "ScreenshotData.h"

#define ONPAINT_TIMEOUT 10
#define ELAPSED_TIME_ONPAINT_TIMEOUT 2500

class ScreenshotDataModule : public DataModuleBase, public CefBaseRefCounted {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotDataModule);
    int height;
    int width;
    bool mobile;
    CefRefPtr<ScreenshotHandler> screenshotHandler;
    CefRefPtr<ScreenshotClient> screenshotClient;
    CefRefPtr<CefBrowser> browser;

public:
    DataBase* process(CefMainArgs*, std::string) OVERRIDE;

    ScreenshotDataModule();
    ScreenshotDataModule(int, int, bool);
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

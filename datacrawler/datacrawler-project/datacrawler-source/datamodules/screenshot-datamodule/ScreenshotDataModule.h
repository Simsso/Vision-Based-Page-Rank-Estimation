//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H
#define DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

#include <include/internal/cef_ptr.h>
#include "../DataModuleBase.h"
#include "ScreenshotHandler.h"
#include "ScreenshotClient.h"


class ScreenshotDataModule : public DataModuleBase {
private:
    int height;
    int width;
    ScreenshotHandler* screenshotHandler;
    CefRefPtr<ScreenshotClient> screenshotClient;
    CefRefPtr<CefBrowser> browser;

public:
    NodeElement* process(string);

    ScreenshotDataModule();
    ScreenshotDataModule(int, int);
    ~ScreenshotDataModule();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTDATAMODULE_H

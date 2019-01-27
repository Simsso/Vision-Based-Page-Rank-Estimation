//
// Created by doktorgibson on 1/27/19.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOTLOADHANDLER_H
#define DATACRAWLER_PROJECT_SCREENSHOTLOADHANDLER_H

#include <include/cef_load_handler.h>
#include "../../util/Logger.h"


class ScreenshotLoadhandler : public CefLoadHandler {
private:
    IMPLEMENT_REFCOUNTING(ScreenshotLoadhandler)
    Logger* logger;
    bool * finishedLoading;

public:
    void OnLoadingStateChange(CefRefPtr<CefBrowser>, bool, bool, bool) OVERRIDE;
    ScreenshotLoadhandler(bool*);
    ScreenshotLoadhandler();
    ~ScreenshotLoadhandler() OVERRIDE;
};


#endif //DATACRAWLER_PROJECT_SCREENSHOTLOADHANDLER_H

//
// Created by doktorgibson on 1/27/19.
//

#include "ScreenshotLoadhandler.h"


ScreenshotLoadhandler::ScreenshotLoadhandler() {
    logger = Logger::getInstance();
}

ScreenshotLoadhandler::ScreenshotLoadhandler(bool * finishedLoading) {
    logger = Logger::getInstance();
    this->finishedLoading = finishedLoading;
}

ScreenshotLoadhandler::~ScreenshotLoadhandler() {}

void ScreenshotLoadhandler::OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                                          bool isLoading,
                                          bool canGoBack,
                                          bool canGoForward) {

    *finishedLoading = !isLoading;
}

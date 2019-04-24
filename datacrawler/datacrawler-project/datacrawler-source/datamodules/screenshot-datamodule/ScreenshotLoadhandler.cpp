//
// Created by doktorgibson on 1/27/19.
//

#include "ScreenshotLoadhandler.h"


ScreenshotLoadhandler::ScreenshotLoadhandler(bool& finishedLoading) : finishedLoading(finishedLoading) {
    logger = Logger::getInstance();
}

ScreenshotLoadhandler::~ScreenshotLoadhandler() {}

void ScreenshotLoadhandler::OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                                          bool isLoading,
                                          bool canGoBack,
                                          bool canGoForward) {

    finishedLoading = !isLoading;
}

//
// Created by samed on 23.10.18.
//

#include <include/cef_app.h>
#include "ScreenshotDataModule.h"

//  TODO clean-up
ScreenshotDataModule::~ScreenshotDataModule() {}

ScreenshotDataModule::ScreenshotDataModule() {}

ScreenshotDataModule::ScreenshotDataModule(int height, int width) {
    this->height = height;
    this->width = width;
}

NodeElement *ScreenshotDataModule::process(string url) {
    logger->info("Running ScreenshotDataModule ..");
    this->url = url;

    screenshotHandler = new ScreenshotHandler(height, width);
    screenshotClient = new ScreenshotClient(screenshotHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    // TODO Detect when website has finished loading
    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 1;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    CefRunMessageLoop();

    NodeElement *tmp = new NodeElement();
    logger->info("Running ScreenshotDataModule .. finished !");
    return tmp;
}
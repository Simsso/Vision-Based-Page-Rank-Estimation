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

    NodeElement * nodeElement = new NodeElement();

    // no mutex needed since MessageLoops is only exited, when painting is over
    screenshotHandler = new ScreenshotHandler(nodeElement, height, width);
    screenshotClient = new ScreenshotClient(screenshotHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    // TODO Detect when website has finished loading
    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 1;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    CefRunMessageLoop();

    logger->info("Running ScreenshotDataModule .. finished !");
    return nodeElement;
}
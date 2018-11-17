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

    NodeElement * nodeElement = nullptr;

    // no mutex needed since MessageLoops is only exited, when painting is over
    screenshotHandler = new ScreenshotHandler(nodeElement, height, width);
    screenshotClient = new ScreenshotClient(screenshotHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    // TODO Detect when website has finished loading
    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 60;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    std::thread timeout([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(SCREENSHOT_TIMEOUT));
        screenshotHandler->getMutex().lock();

        if(screenshotHandler->hasPainted()){
            logger->info("ScreenshotDataModule timed out! Returning current screenshot result!");
        } else {
            // TODO Add placeholder screenshot
            logger->error("ScreenshotDataModule has failed to take a screenshot! Returning placeholder!");
            throw "ScreenshotDataModule has failed to take a screenshot!";
        }

        CefQuitMessageLoop();
        screenshotModuleMutex.unlock();
    });


    CefRunMessageLoop();

    logger->info("Running ScreenshotDataModule .. finished !");
    return nodeElement;
}
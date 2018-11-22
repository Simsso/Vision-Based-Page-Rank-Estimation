#include <include/cef_app.h>
#include "ScreenshotDataModule.h"

/**
 * ~ScreenshotDataModule
 */
ScreenshotDataModule::~ScreenshotDataModule() {
    delete screenshotHandler;
    delete screenshotClient;
}

/**
 * ScreenshotDataModule
 */
ScreenshotDataModule::ScreenshotDataModule() {}

/**
 * ScreenshotDataModule - Initialized ScreenshotDataModule
 * @param height represents the height of the screenshot to take
 * @param width represents the width of the screenshot to take
 */
ScreenshotDataModule::ScreenshotDataModule(int height, int width) {
    this->height = height;
    this->width = width;
}

/**
 * process - Takes screenshot for a given url
 * @param url represents the website, which shall be visited and screenshot taken of
 * @return Database, which represents a single DataModule in the graph
 */
DataBase *ScreenshotDataModule::process(std::string url) {
    logger->info("Running ScreenshotDataModule ..");
    this->url = url;

    bool * quitMessageLoop = new bool;
    *quitMessageLoop = false;

    // no mutex needed since MessageLoops is only exited, when painting is over
    screenshotHandler = new ScreenshotHandler(quitMessageLoop, 5, 0.05, height, width);
    screenshotClient = new ScreenshotClient(screenshotHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 60;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    // Thread to timeout ScreenshotHandler::onPaint(), if detecting mechanisms fail
    std::thread timeout([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(ONPAINT_TIMEOUT));

        if (screenshotHandler->hasPainted()) {
            logger->info("ScreenshotDataModule timed out! Returning current screenshot result!");
        } else {
            logger->error("ScreenshotDataModule has failed to take a screenshot!");
            throw "ScreenshotDataModule has failed to take a screenshot!";
        }

        screenshotHandler->getQuitMessageLoopMutex().lock();
        *quitMessageLoop = true;
        screenshotHandler->getQuitMessageLoopMutex().unlock();
    });

    // Thread to check, whether ScreenshotHandler::onPaint() was called in the last ELAPSED_TIME_ONPAINT_TIMEOUT
    std::thread screenshotHandlerStopped([&]() {

        while (!screenshotHandler->hasPainted());

        while (screenshotHandler->getTimeSinceLastPaint() < ELAPSED_TIME_ONPAINT_TIMEOUT);

        logger->info(std::to_string(ELAPSED_TIME_ONPAINT_TIMEOUT) +
                     "ms has passed since last OnPaint()! Taking screenshot!");

        screenshotHandler->getQuitMessageLoopMutex().lock();
        *quitMessageLoop = true;
        screenshotHandler->getQuitMessageLoopMutex().unlock();
    });

    while (!(*quitMessageLoop)){
        CefDoMessageLoopWork();
    }

    ScreenshotData *screenshotData = new ScreenshotData();
    screenshotData->addScreenshot(screenshotHandler->getScreenshot(), height, width);

    logger->info("Waiting for all threads to terminate ..");
    screenshotHandlerStopped.join();
    timeout.join();
    logger->info("Running ScreenshotDataModule .. finished !");


    return screenshotData;
}
#include "ScreenshotDataModule.h"

/**
 * ~ScreenshotDataModule
 */
ScreenshotDataModule::~ScreenshotDataModule() {}

/**
 * ScreenshotDataModule
 */
ScreenshotDataModule::ScreenshotDataModule() {}

/**
 * ScreenshotDataModule - Initialized ScreenshotDataModule
 * @param height represents the height of the screenshot to take
 * @param width represents the width of the screenshot to take
 */
ScreenshotDataModule::ScreenshotDataModule(int height, int width, int onPaintTimeout, int elapsedTimeOnPaintTimeout, double changeThreshold, int lastScreenshots, bool mobile) {
    this->height = height;
    this->width = width;
    this->mobile = mobile;
    this->onPaintTimeout = onPaintTimeout;
    this->elapsedTimeOnPaintTimeout = elapsedTimeOnPaintTimeout;
    this->changeThreshold = changeThreshold;
    this->lastScreenshots = lastScreenshots;
}

/**
 * process - Takes screenshot for a given url
 * @param url represents the website, which shall be visited and screenshot taken of
 * @return Database, which represents a single DataModule in the graph
 */
DataBase *ScreenshotDataModule::process(std::string url) {

    std::map<std::string, std::string> map;

    if(mobile) {
        logger->info("Running ScreenshotMobile-DataModule ..");
        logger->info("Switching to mobile user-agent!");
        map.insert(std::pair<std::string, std::string>("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25"));
    }
    else
        logger->info("Running Screenshot-DataModule ..");

    logger->info("Requiring UI Thread ..");
    CEF_REQUIRE_UI_THREAD();

    if(!CefCurrentlyOn(TID_UI))
        throw "Not in UI Thread!";

    logger->info("Runnning in UI thread!");

    bool * quitMessageLoop = new bool;
    *quitMessageLoop = false;

    CefRefPtr<ScreenshotRequestHandler> screenshotRequestHandler(new ScreenshotRequestHandler(map));
    CefRefPtr<ScreenshotHandler> screenshotHandler(new ScreenshotHandler(quitMessageLoop, lastScreenshots, changeThreshold, height, width));
    CefRefPtr<ScreenshotClient> screenshotClient(new ScreenshotClient(screenshotHandler, screenshotRequestHandler));

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 60;

    CefBrowserHost::CreateBrowser(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    // Thread to timeout ScreenshotHandler::onPaint(), if detecting mechanisms fail
    std::thread timeout([&]() {
        int secondsSteps = 0;

        while(secondsSteps < onPaintTimeout){
            if(*quitMessageLoop)
                return;

            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++secondsSteps;
        }

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

    // Thread to check, whether ScreenshotHandler::onPaint() was called in the last elapsedTimeOnPaintTimeout
    std::thread screenshotHandlerStopped([&]() {

        while (!screenshotHandler->hasPainted());

        while (screenshotHandler->getTimeSinceLastPaint() < elapsedTimeOnPaintTimeout){
            if(*quitMessageLoop)
                return;
        }

        logger->info(std::to_string(elapsedTimeOnPaintTimeout) +
                     "ms has passed since last OnPaint()! Taking screenshot!");

        screenshotHandler->getQuitMessageLoopMutex().lock();
        *quitMessageLoop = true;
        screenshotHandler->getQuitMessageLoopMutex().unlock();
    });

    while (!(*quitMessageLoop)){
        CefDoMessageLoopWork();
    }

    Screenshot *screenshot = new Screenshot(screenshotHandler->getScreenshot(), height, width, mobile);

    logger->info("Waiting for all threads to terminate ..");
    screenshotHandlerStopped.join();
    timeout.join();

    logger->info("Running Screenshot-DataModule .. finished !");

    return screenshot;
}
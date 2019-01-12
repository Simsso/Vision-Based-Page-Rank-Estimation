#include "ScreenshotDataModule.h"

/**
 * ~ScreenshotDataModule
 */
ScreenshotDataModule::~ScreenshotDataModule() {
    delete screenshotHandler;
    delete screenshotClient;
    delete screenshotRequestHandler;
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
ScreenshotDataModule::ScreenshotDataModule(int height, int width, int ONPAINT_TIMEOUT, int ELAPSED_TIME_ONPAINT_TIMEOUT, double CHANGE_THRESHOLD, int LAST_SCREENSHOTS, bool mobile) {
    this->height = height;
    this->width = width;
    this->mobile = mobile;
    this->ONPAINT_TIMEOUT = ONPAINT_TIMEOUT;
    this->ELAPSED_TIME_ONPAINT_TIMEOUT = ELAPSED_TIME_ONPAINT_TIMEOUT;
    this->CHANGE_THRESHOLD = CHANGE_THRESHOLD;
    this->LAST_SCREENSHOTS = LAST_SCREENSHOTS;
}

/**
 * process - Takes screenshot for a given url
 * @param url represents the website, which shall be visited and screenshot taken of
 * @return Database, which represents a single DataModule in the graph
 */
DataBase *ScreenshotDataModule::process(CefMainArgs* mainArgs, std::string url) {
    logger->info("Running ScreenshotDataModule ..");

    CefSettings cefSettings;

    CefInitialize(*mainArgs, cefSettings, NULL, NULL);
    logger->info("Initializing CEF finished .. !");

    logger->info("Requiring UI Thread for Screenshot-DataModule ..");
    CEF_REQUIRE_UI_THREAD();

    if(!CefCurrentlyOn(TID_UI))
        throw "Screenshot-Datamodule not in UI Thread!";

    logger->info("Runnning in UI thread!");

    bool * quitMessageLoop = new bool;
    *quitMessageLoop = false;

    std::map<std::string, std::string> map;

    if(mobile)
        map.insert(std::pair<std::string, std::string>("User-Agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 6_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/6.0 Mobile/10A5376e Safari/8536.25"));

    screenshotRequestHandler = new ScreenshotRequestHandler(map);
    screenshotHandler = new ScreenshotHandler(quitMessageLoop, LAST_SCREENSHOTS, CHANGE_THRESHOLD, height, width);
    screenshotClient = new ScreenshotClient(screenshotHandler, screenshotRequestHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 60;

    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    // Thread to timeout ScreenshotHandler::onPaint(), if detecting mechanisms fail
    std::thread timeout([&]() {
        int secondsSteps = 0;

        while(secondsSteps < ONPAINT_TIMEOUT){
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

    // Thread to check, whether ScreenshotHandler::onPaint() was called in the last ELAPSED_TIME_ONPAINT_TIMEOUT
    std::thread screenshotHandlerStopped([&]() {

        while (!screenshotHandler->hasPainted());

        while (screenshotHandler->getTimeSinceLastPaint() < ELAPSED_TIME_ONPAINT_TIMEOUT){
            if(*quitMessageLoop)
                return;
        }

        logger->info(std::to_string(ELAPSED_TIME_ONPAINT_TIMEOUT) +
                     "ms has passed since last OnPaint()! Taking screenshot!");

        screenshotHandler->getQuitMessageLoopMutex().lock();
        *quitMessageLoop = true;
        screenshotHandler->getQuitMessageLoopMutex().unlock();
    });

    while (!(*quitMessageLoop)){
        CefDoMessageLoopWork();
    }

    ScreenshotData *screenshotData = new ScreenshotData(mobile);
    screenshotData->addScreenshot(screenshotHandler->getScreenshot(), height, width);

    logger->info("Waiting for all threads to terminate ..");
    screenshotHandlerStopped.join();
    timeout.join();

    logger->info("Running ScreenshotDataModule .. finished !");

    browser.get()->Release();

    CefShutdown();
    logger->info("Shut down CEF!");

    return screenshotData;
}
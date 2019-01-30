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
ScreenshotDataModule::ScreenshotDataModule(int height, int width, int onPaintTimeout, int elapsedTimeOnPaintTimeout, bool mobile) {
    this->height = height;
    this->width = width;
    this->mobile = mobile;
    this->onPaintTimeout = onPaintTimeout;
    this->elapsedTimeOnPaintTimeout = elapsedTimeOnPaintTimeout;
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

    std::unique_ptr<bool> quitMessageLoop(new bool(false));
    std::unique_ptr<bool> browserFinishedLoading(new bool(false));

    CefRefPtr<ScreenshotLoadhandler> screenshotLoadhandler(new ScreenshotLoadhandler(browserFinishedLoading.get()));
    CefRefPtr<ScreenshotRequestHandler> screenshotRequestHandler(new ScreenshotRequestHandler(map));
    CefRefPtr<ScreenshotHandler> screenshotHandler(new ScreenshotHandler(height, width, quitMessageLoop.get()));
    CefRefPtr<ScreenshotClient> screenshotClient(new ScreenshotClient(screenshotHandler, screenshotRequestHandler, screenshotLoadhandler));

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;
    browserSettings.windowless_frame_rate = 1;

    CefRefPtr <CefBrowser> browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, screenshotClient.get(), url, browserSettings, NULL);

    std::thread timeout([&]() {
        int secondsSteps = 0;

        while(secondsSteps < elapsedTimeOnPaintTimeout){
           if(*quitMessageLoop)
                return;

            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++secondsSteps;
        }

        logger->info("ScreenshotDataModule timed out after "+std::to_string(elapsedTimeOnPaintTimeout)+" seconds! Returning current screenshot result!");

        *quitMessageLoop = true;
    });

    std::thread screenshotHandlerStopped([&]() {

        while(!*browserFinishedLoading){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if(*quitMessageLoop)
                return;
        }

        int secondsSteps = 0;

        while(secondsSteps < onPaintTimeout){
            if(*quitMessageLoop)
                return;

            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++secondsSteps;
        }

        logger->info("Waited "+std::to_string(onPaintTimeout)+" seconds for rendering ..");

        *quitMessageLoop = true;
    });

    while (!(*quitMessageLoop)){
        CefDoMessageLoopWork();
    }

    Screenshot *screenshot = new Screenshot(screenshotHandler->getScreenshot(), height, width, mobile);

    logger->info("Waiting for all threads to terminate ..");
    screenshotHandlerStopped.join();
    timeout.join();

    logger->info("Running Screenshot-DataModule .. finished !");

    browser->GetHost()->CloseBrowser(true);

    return screenshot;
}
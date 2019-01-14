#include "datacrawler.h"

Logger *Logger::instance = 0;

#include "opencv2/opencv.hpp"
#include "stdlib.h"
#include "DatacrawlerApp.h"

int main(int argc, char *argv[]) {
    setenv("LOG_LEVEL", "LOG_ALL", true);
    Logger *logger = Logger::getInstance();

    CefMainArgs mainArgs(argc, argv);

    CefRefPtr<DatacrawlerApp> datacrawlerApp(new DatacrawlerApp());

    int exitCode = CefExecuteProcess(mainArgs, datacrawlerApp, NULL);
    if (exitCode >= 0) {
        // The sub-process terminated, exit now.
        return exitCode;
    }

    CefSettings cefSettings;
    cefSettings.windowless_rendering_enabled = true;

    if(CefInitialize(mainArgs, cefSettings, NULL, NULL))
        logger->info("Initializing CEF finished .. !");
    else {
        logger->fatal("Initializing has failed!");
    }

    Datacrawler datacrawler;
    datacrawler.init();

    string url = "sap.com";
   /* NodeElement * node =*/ datacrawler.process(url);

    /* string urlRank = "1";
    string outputPath = "/home/Desktop/";

    vector<DataBase*>* dataBase = node->getData();

    for(auto x: *dataBase){
        DataModulesEnum dataModulesEnum = x->getDataModuleType();

        if(dataModulesEnum == SCREENSHOT_MODULE || dataModulesEnum == SCREENSHOT_MOBILE_MODULE) {

            auto screenshotList = (ScreenshotData*) x;

            for(auto img: screenshotList->getScreenshots())
            try {
                std::vector<int> params;
                params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                params.push_back(9);
                cv::Mat newImg = cv::Mat(img->getHeight(),img->getWidth(), CV_8UC4, img->getScreenshot());

                std::string fullPath;
                fullPath.append(outputPath);
                fullPath.append("/");
                fullPath.append(urlRank);
                fullPath.append(".png");
                logger->info("We have screenshots! Saving to to: "+fullPath);
                cv::imwrite(fullPath, newImg, params);

            } catch(std::runtime_error& ex) {
                fprintf(stderr, "Exception while converting taking picture of the website in PNG format: %s", ex.what());
            }
        } else {
            logger->info("No data has been produced!");
        }
    }
 */
    logger->info("Datacrawler execution finished!");
    CefShutdown();
}
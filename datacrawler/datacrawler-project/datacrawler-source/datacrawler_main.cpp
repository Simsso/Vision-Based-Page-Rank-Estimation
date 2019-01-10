#include "datacrawler.h"

#define DATASET_VERSION_1

#define DEVELOPMENT true

Logger *Logger::instance = 0;

#ifdef DATASET_VERSION_1

#include "opencv2/opencv.hpp"
#include "stdlib.h"

int main(int argc, char *argv[]) {

    if(DEVELOPMENT){
        setenv("LOG_LEVEL", "LOG_ALL", true);
        setenv("URL","http://youtube.com", true);
        setenv("DATAMODULE", "SCREENSHOT_MODULE", true);
        setenv("URL_RANK", "1", true);
        setenv("OUTPUT_PATH", "/home/doktorgibson/Desktop/", true);
        setenv("ONPAINT_TIMEOUT", "25", true);
        setenv("ELAPSED_TIME_ONPAINT_TIMEOUT", "17500", true);
        setenv("CHANGE_THRESHOLD", "0.005", true);
        setenv("LAST_SCREENSHOTS", "20", true);
    }

    Logger *logger = Logger::getInstance();

    char *url = std::getenv("URL");
    if(url == NULL){
        logger->fatal("URL not specified! Exiting!");
        exit(1);
    }

    CefMainArgs mainArgs(argc, argv);
    CefExecuteProcess(mainArgs, NULL, NULL);

    logger->info("Starting Datacrawler !");

    Datacrawler datacrawler(&mainArgs);

    datacrawler.init();

    NodeElement * node = datacrawler.process(url);

    char *urlRank = getenv("URL_RANK");
    char *outputPath = getenv("OUTPUT_PATH");

    if(urlRank == NULL){
        logger->fatal("URL_RANK not specified! Exiting!");
        exit(1);
    }

    if(outputPath == NULL){
        logger->fatal("OUTPUT_PATH not specified! Exiting!");
        exit(1);
    }

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

    logger->info("Datacrawler execution finished!");
}

#endif // DATASET_VERSION_1

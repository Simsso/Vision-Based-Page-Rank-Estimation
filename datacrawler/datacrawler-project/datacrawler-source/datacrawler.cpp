#include "datacrawler.h"

/**
 * Datacrawler
 */
Datacrawler::Datacrawler(CefMainArgs *mainArgs) {
    logger = Logger::getInstance();
    this->mainArgs = mainArgs;
}

Datacrawler::~Datacrawler() {}

/**
 * init - Loads all user-defined DataModules and prepares Datacrawler to crawl given url
 */
void Datacrawler::init() {
    logger->info("Initialising Datacrawler !");

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE)->createInstance());
        logger->info("Using ScreenshotDataModule ..");
    }

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE)->createInstance());
        logger->info("Using ScreenshotDataModule with mobile enabled ..");
    }

    if (datacrawlerConfiguration.getConfiguration(URL_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(URL_MODULE)->createInstance());
        logger->info("Using URLDataModule ..");
    }

    logger->info("Initialising Datacrawler finished!");
}

/**
 * process - Process given url with loaded DataModules
 * @param url which should be processed
 * @return NodeElement which represents a node in the graph with all data the user defined for the graph
 */
NodeElement *Datacrawler::process(string url) {
    logger->info("Processing <" + url + ">");
    logger->info("Running DataModules!");

    NodeElement *newNode = new NodeElement();

    for (auto x: dataModules) {
                    x->process(mainArgs, url);
    }

    logger->info("<" + url + "> processed!");

    return newNode;
}
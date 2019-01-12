#include "datacrawler.h"

/**
 * Datacrawler
 */
Datacrawler::Datacrawler() {
    logger = Logger::getInstance();
}

Datacrawler::~Datacrawler() {
    for (auto x: dataModules) {
        delete x;
    }
}

/**
 * init - Loads all user-defined DataModules and prepares Datacrawler to crawl given url
 */
void Datacrawler::init() {
    logger->info("Initialising Datacrawler !");

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE)->createInstance());
        logger->info("Using Screenshot-DataModule ..");
    }

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE)->createInstance());
        logger->info("Using ScreenshotMobile-DataModule ..");
    }

    if (datacrawlerConfiguration.getConfiguration(URL_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(URL_MODULE)->createInstance());
        logger->info("Using URL-Module ..");
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
       newNode->addData(x->process(url));
    }

    logger->info("<" + url + "> processed!");

    return newNode;
}
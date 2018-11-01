//
// Created by samed on 23.10.18.
//

#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>
#include "datacrawler.h"


Datacrawler::Datacrawler(){
    logger = Logger::getInstance();
}

Datacrawler::~Datacrawler(){}

void Datacrawler::init() {
        logger->info("Initialising Datacrawler !");

        if(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE) != nullptr){
            dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE)->createInstance());
            logger->info("Using ScreenshotDataModule ..");
        }

        if(datacrawlerConfiguration.getConfiguration(URL_MODULE) != nullptr){
            dataModules.push_front(datacrawlerConfiguration.getConfiguration(URL_MODULE)->createInstance());
            logger->info("Using URLDataModule ..");
        }

        logger->info("Initialising Datacrawler finished!");
}

NodeElement* Datacrawler::process(string url) {
        logger->info("Processing <"+url+">");
        logger->info("Running DataModules!");

        for (auto x: dataModules) {
            x->process(url);
        }

        logger->info("<"+url+"> processed!");

    return nullptr;
}
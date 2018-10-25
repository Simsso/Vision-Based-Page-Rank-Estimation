//
// Created by samed on 23.10.18.
//

#include "ScreenshotDataModule.h"

ScreenshotDataModule::ScreenshotDataModule(){}
ScreenshotDataModule::~ScreenshotDataModule(){}

NodeElement* ScreenshotDataModule::process() {

    logger->info("Processing ScreenshotDataModule ..");
    NodeElement* tmp = new NodeElement();
    logger->info("Processing ScreenshotDataModule .. finished !");
    return tmp;
}
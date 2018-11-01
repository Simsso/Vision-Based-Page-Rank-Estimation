//
// Created by samed on 23.10.18.
//

#include "ScreenshotDataModule.h"
ScreenshotDataModule::~ScreenshotDataModule() {}

ScreenshotDataModule::ScreenshotDataModule() {}

ScreenshotDataModule::ScreenshotDataModule(int height, int width) {
    this->height = height;
    this->width = width;
}

NodeElement *ScreenshotDataModule::process(string url) {
    logger->info("Running ScreenshotDataModule ..");
    this->url = url;


    NodeElement *tmp = new NodeElement();
    logger->info("Running ScreenshotDataModule .. finished !");
    return tmp;
}
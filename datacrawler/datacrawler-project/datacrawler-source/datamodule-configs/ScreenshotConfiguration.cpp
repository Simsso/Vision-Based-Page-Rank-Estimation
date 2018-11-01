//
// Created by samed on 23.10.18.
//

#include "ScreenshotConfiguration.h"

ScreenshotConfiguration::ScreenshotConfiguration(){
    width = 800;
    height = 600;
}

ScreenshotConfiguration::ScreenshotConfiguration(int width, int height){
    this->width = width;
    this->height = height;
}

ScreenshotConfiguration::~ScreenshotConfiguration(){}

DataModuleBase* ScreenshotConfiguration::createInstance() {
    return new ScreenshotDataModule(height, width);
}


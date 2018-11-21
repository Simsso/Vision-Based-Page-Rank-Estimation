//
// Created by samed on 01.11.18.
//

#include "ScreenshotHandler.h"

ScreenshotHandler::~ScreenshotHandler() {

}

ScreenshotHandler::ScreenshotHandler() {
    renderHeight = 600;
    renderWidth = 800;
    logger = Logger::getInstance();
    lastScreenshot = nullptr;
    mHasPainted = false;
    numInvokations = 0;
    sumL1Distance = 0;
    initialInvoke = true;

    for(int i = 0; i < 4; i++)
        averageL1Distances[i] = 0;
}

ScreenshotHandler::ScreenshotHandler(NodeElement* nodeElement, int renderHeight, int renderWidth) {
    this->nodeElement = nodeElement;
    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;
    logger = Logger::getInstance();
    lastScreenshot = nullptr;
    mHasPainted = false;
    numInvokations = 0;
    sumL1Distance = 0;
    initialInvoke = true;

    for(int i = 0; i < 4; i++)
        averageL1Distances[i] = 0;
}

bool ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderHeight, renderWidth);
    return true;
}

void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
        const void *buffer, int width, int height) {
    screenshotModuleMutex.lock();
    //logger->info("Painting!");
    unsigned char* screenshot = (unsigned char*) buffer;
    mHasPainted = true;

    int32_t deltaNorm = 0;

    // TODO  Quit, when last onPaint() is older than 1000m
    // TODO  Calculate a change matrix (consisting of 1 and 0) and calculate the L1 distance
    if (!initialInvoke){
        deltaNorm = calculateL1Distance(lastScreenshot, screenshot, width, height);
        delete lastScreenshot;
        ++numInvokations;

        sumL1Distance += deltaNorm;
        insertAverageL1Distance(sumL1Distance /  numInvokations);

        if(averageL1Distances[0] != 0){
            changeRatesL1distances.push_back(calculateChangeRate());
            logger->info("Sum: "+std::to_string(sumL1Distance)+" Invokations: "+std::to_string(numInvokations)+
        +" Current L1-Distance: "+ std::to_string(deltaNorm)+" Avg: "+std::to_string(averageL1Distances[3])+" Average change-rate: "+std::to_string(changeRatesL1distances.back()));

            if( abs(1 - (changeRatesL1distances.at(0) / changeRatesL1distances.back())) <= 0.05){
                logger->info("Change-rate dropped is under 5% of the initial change-rate!");
            }
        }


    }

    nodeElement = nullptr;

    lastScreenshot = new unsigned char[height * width *4];
    memcpy(lastScreenshot, buffer, sizeof(unsigned char) * height * width * 4);
    screenshotModuleMutex.unlock();
    initialInvoke = false;
}

// TODO Refactor the function to calculate the change rate
 double ScreenshotHandler::calculateChangeRate(){
    return (double) (-2 * averageL1Distances[0] + 9 * averageL1Distances[1] - 18 * averageL1Distances[2] + 11 * averageL1Distances[3]) / (double) 6;
}

 int32_t ScreenshotHandler::calculateL1Distance(unsigned char* firstMatrix, unsigned char* secMatrix, int32_t numCol, int32_t numRow) {
    int32_t l1 = 0;

    // our matrix is an array each pixel is represented by 4 * 2 Bytes due BRGA
    // if we take a screenshot of 640*420 = 268800 pixels, we have to consider 1075200 pixels
    for(int32_t i = 0; i < numRow*numCol * 4; i++ )
            l1 +=  abs((int32_t) *(firstMatrix + i) - (int32_t) *(secMatrix + i));

    return l1;
}

 void ScreenshotHandler::insertAverageL1Distance(int32_t value){

    for(int i = 0; i < 3; i++) {
        averageL1Distances[i] = averageL1Distances[i+1];
    }
    averageL1Distances[3] = value;
}

bool ScreenshotHandler::hasPainted(){ return mHasPainted;}

std::mutex& ScreenshotHandler::getMutex() { return screenshotModuleMutex;}
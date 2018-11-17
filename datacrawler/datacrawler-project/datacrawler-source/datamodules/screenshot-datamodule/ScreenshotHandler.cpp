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
}

ScreenshotHandler::ScreenshotHandler(NodeElement* nodeElement, int renderHeight, int renderWidth) {
    this->nodeElement = nodeElement;
    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;
    logger = Logger::getInstance();
    lastScreenshot = nullptr;
    mHasPainted = false;
    num =0;
    average = 0;
    sum = 0;
}

bool ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderHeight, renderWidth);
    return true;
}

void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
        const void *buffer, int width, int height) {
    screenshotModuleMutex.lock();
    logger->info("Painting!");
    unsigned char* screenshot = (unsigned char*) buffer;
    mHasPainted = true;

    int32_t deltaNorm = 0;

    if (lastScreenshot != nullptr){
        deltaNorm = calculateL1Distance(lastScreenshot, screenshot, width, height);
        logger->info("L1 distance between screenshots: "+std::to_string(deltaNorm));
        delete lastScreenshot;
        ++num;
        sum += deltaNorm;

        average = sum /  num;
        logger->info("Num: "+std::to_string(num)+" Avg: "+std::to_string(average));
    }

    nodeElement = nullptr;

   /* if(deltaNorm == 0)
        CefQuitMessageLoop();
    else */

    lastScreenshot = new unsigned char[height * width *4];
    memcpy(lastScreenshot, buffer, sizeof(unsigned char) * height * width * 4);
    screenshotModuleMutex.unlock();
}

 int32_t ScreenshotHandler::calculateL1Distance(unsigned char* firstMatrix, unsigned char* secMatrix, int32_t numCol, int32_t numRow) {
    int32_t l1 = 0;

    // our matrix is an array each pixel is represented by 4 * 2 Bytes due BRGA
    // if we take a screenshot of 640*420 = 268800 pixels, we have to consider 1075200 pixels
    for(int32_t i = 0; i < numRow*numCol * 4; i++ )
            l1 +=  abs((int32_t) *(firstMatrix + i) - (int32_t) *(secMatrix + i));

    return l1;
}

bool ScreenshotHandler::hasPainted(){ return mHasPainted;}

std::mutex& ScreenshotHandler::getMutex() { return screenshotModuleMutex;}
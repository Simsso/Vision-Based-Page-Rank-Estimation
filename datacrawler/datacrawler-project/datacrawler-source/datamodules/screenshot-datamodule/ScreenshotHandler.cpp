//
// Created by samed on 01.11.18.
//

#include "ScreenshotHandler.h"

ScreenshotHandler::~ScreenshotHandler() {

}

ScreenshotHandler::ScreenshotHandler() {
    renderHeight = 600;
    renderWidth = 800;
    deltaNorm = 1;
    logger = Logger::getInstance();

}

ScreenshotHandler::ScreenshotHandler(NodeElement* nodeElement, int renderHeight, int renderWidth) {
    this->nodeElement = nodeElement;
    this->renderHeight = renderHeight;
    this->renderWidth = renderWidth;
    deltaNorm = 1;
    logger = Logger::getInstance();

}

bool ScreenshotHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, renderHeight, renderWidth);
    return true;
}

void ScreenshotHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
        const void *buffer, int width, int height) {


    //milliseconds invokeTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    //invokes.push_back(invokeTime.count());

    logger->info("Painting!");
    //logger->info(std::to_string(invokes.front()));
    auto norm = calculateL1Norm(buffer, width, height);
    deltaNorm = abs(lastL1Norm - norm);

    logger->info("L1 distance between screenshots: "+std::to_string(deltaNorm));

    nodeElement = nullptr;

   /* if(deltaNorm == 0)
        CefQuitMessageLoop();
    else */
        lastL1Norm = norm;

}

 int32_t ScreenshotHandler::calculateL1Norm(const void* mat, int32_t  numCol, int32_t numRow) {
    unsigned char * matrix = (unsigned char *) mat;
    int32_t l1 = 0;

     // our matrix is an array each pixel is represented by 4 * 2 Bytes due BRGA
     // if we take a screenshot of 640*420 = 268800 pixels, we have to consider 1075200 pixels
    for(int32_t i = 0; i < numRow*numCol * 4; i++ )
            l1 +=  (int32_t) *(matrix + i);

    return l1;
}
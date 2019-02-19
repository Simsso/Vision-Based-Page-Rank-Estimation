#include "Screenshot.h"

/**
 * ~Screenshot
 */
Screenshot::~Screenshot() {
    delete screenshot;
}

/**
 * Screenshot - Represents a single screenshot taken by ScreenshotDataModule
 * @param screenshot the screenshot, which shall be wrapped in this class
 * @param height of the screenshot
 * @param width of the screenshot
 */
Screenshot::Screenshot(unsigned char * screenshot, int height, int width) {
    this->screenshot = screenshot;
    this->height = height;
    this->width = width;
}

/**
 * getScreenshot
 * @return screenshot in this class
 */
unsigned char* Screenshot::getScreenshot() { return screenshot;}

int Screenshot::getHeight() { return height;}

int Screenshot::getWidth() { return width;}


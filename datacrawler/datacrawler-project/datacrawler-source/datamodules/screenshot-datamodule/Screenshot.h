//
// Created by doktorgibson on 11/22/18.
//

#ifndef DATACRAWLER_PROJECT_SCREENSHOT_H
#define DATACRAWLER_PROJECT_SCREENSHOT_H


class Screenshot {
private:
    unsigned char* screenshot;
    int height;
    int width;
public:

    unsigned char* getScreenshot();
    int getHeight();
    int getWidth();

    Screenshot(unsigned char*, int, int);
    ~Screenshot();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOT_H

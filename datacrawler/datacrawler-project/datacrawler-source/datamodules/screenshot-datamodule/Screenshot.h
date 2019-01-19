#ifndef DATACRAWLER_PROJECT_SCREENSHOT_H
#define DATACRAWLER_PROJECT_SCREENSHOT_H


#include "../DataBase.h"

class Screenshot : public DataBase {
private:
    unsigned char* screenshot;
    int height;
    int width;
    bool mobile;

public:
    DataModulesEnum getDataModuleType();
    unsigned char* getScreenshot();
    int getHeight();
    int getWidth();
    bool isMobile();

    Screenshot(unsigned char*, int, int, bool);
    ~Screenshot();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOT_H

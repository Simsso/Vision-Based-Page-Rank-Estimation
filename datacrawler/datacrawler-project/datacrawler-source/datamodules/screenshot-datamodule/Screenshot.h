#ifndef DATACRAWLER_PROJECT_SCREENSHOT_H
#define DATACRAWLER_PROJECT_SCREENSHOT_H


#include "../DataBase.h"

class Screenshot : public DataBase {
private:
    int height;
    int width;
    bool mobile;
    unsigned char * screenshot;

public:
    unsigned char* getScreenshot();
    int getHeight();
    int getWidth();
    DataModulesEnum getDataModules();

    Screenshot(unsigned char*, int, int, bool);
    ~Screenshot();
};


#endif //DATACRAWLER_PROJECT_SCREENSHOT_H

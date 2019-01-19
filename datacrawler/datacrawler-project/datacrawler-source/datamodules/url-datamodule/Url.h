//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URL_H
#define DATACRAWLER_PROJECT_URL_H

#include "iostream"
#include "../DataBase.h"

class Url {
private:
    std::string urlText;
    std::string url;
    bool https;

public:
    std::string getUrlText();
    std::string getUrl();
    bool isHttps();

    Url(std::string, std::string, bool);
    ~Url();
};


#endif //DATACRAWLER_PROJECT_URL_H

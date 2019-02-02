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

public:
    std::string getUrlText();
    std::string getUrl();

    Url(const Url&);
    Url(std::string, std::string);
    ~Url();
};


#endif //DATACRAWLER_PROJECT_URL_H

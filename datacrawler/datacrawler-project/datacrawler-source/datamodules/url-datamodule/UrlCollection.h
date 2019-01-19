//
// Created by doktorgibson on 1/19/19.
//

#ifndef DATACRAWLER_PROJECT_URLCOLLECTION_H
#define DATACRAWLER_PROJECT_URLCOLLECTION_H

#include "vector"

#include "../DataBase.h"
#include "Url.h"

class UrlCollection : public DataBase {
private:
    std::vector<Url*>* urls;
    std::string baseUrl;

public:
    DataModulesEnum getDataModuleType();
    void addUrl(Url*);
    std::vector<Url*>* getUrls();
    std::string getBaseUrl();

    UrlCollection(std::string);
    ~UrlCollection();
};


#endif //DATACRAWLER_PROJECT_URLCOLLECTION_H

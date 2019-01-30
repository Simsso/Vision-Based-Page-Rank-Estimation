//
// Created by doktorgibson on 1/19/19.
//

#ifndef DATACRAWLER_PROJECT_URLCOLLECTION_H
#define DATACRAWLER_PROJECT_URLCOLLECTION_H

#include "list"

#include "../DataBase.h"
#include "Url.h"

class UrlCollection : public DataBase {
private:
    std::list<Url>* urls;
    std::string baseUrl;
    bool baseUrlHttps;
    int httpResponseCode;
    std::string clientErrorText;
    int loadingTime;
    size_t size;
    std::string title;

public:
    std::string getTitle();

    void setTitle(std::string title);

    DataModulesEnum getDataModules();

    size_t getSize();
    void setSize(size_t size);

    int getLoadingTime();
    void setLoadingTime(int loadingTime);

    const std::string getClientErrorText();
    void setClientErrorText(std::string clientErrorText);

    int getHttpResponseCode();

    void setHttpResponseCode(int httpResponseCode);
    std::string getBaseUrl();
    void setBaseUrl(std::string baseUrl);
    void setBaseUrlHttps(bool baseUrlHttps);

    void addUrl(std::string, std::string);
    std::list<Url>* getUrls();

    bool isHttps();

    UrlCollection();
    ~UrlCollection();
};


#endif //DATACRAWLER_PROJECT_URLCOLLECTION_H

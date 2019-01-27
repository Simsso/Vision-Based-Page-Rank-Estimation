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
    bool baseUrlHttps;
    int httpResponseCode;
    std::string clientErrorText;
    int loadingTime;
public:
    int getLoadingTime() const;

    void setLoadingTime(int loadingTime);

public:
    const std::string &getClientErrorText() const;

    void setClientErrorText(const std::string &clientErrorText);

public:
    int getHttpResponseCode() const;
    void setHttpResponseCode(int httpResponseCode);
    void setBaseUrl(const std::string &baseUrl);
    void setBaseUrlHttps(bool baseUrlHttps);

    DataModulesEnum getDataModuleType();
    void addUrl(Url*);
    std::vector<Url*>* getUrls();
    std::string getBaseUrl();
    bool isHttps();

    UrlCollection();
    ~UrlCollection();
};


#endif //DATACRAWLER_PROJECT_URLCOLLECTION_H

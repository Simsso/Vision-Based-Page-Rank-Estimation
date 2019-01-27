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
    size_t size;
public:
    size_t getSize() const;
    void setSize(size_t size);

    int getLoadingTime() const;
    void setLoadingTime(int loadingTime);

    const std::string &getClientErrorText() const;
    void setClientErrorText(const std::string &clientErrorText);

    int getHttpResponseCode() const;

    void setHttpResponseCode(int httpResponseCode);
    std::string getBaseUrl();
    void setBaseUrl(const std::string &baseUrl);
    void setBaseUrlHttps(bool baseUrlHttps);

    DataModulesEnum getDataModuleType();
    void addUrl(Url*);
    std::vector<Url*>* getUrls();

    bool isHttps();

    UrlCollection();
    ~UrlCollection();
};


#endif //DATACRAWLER_PROJECT_URLCOLLECTION_H

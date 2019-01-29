//
// Created by doktorgibson on 1/19/19.
//

#include "UrlCollection.h"

UrlCollection::UrlCollection() {
    this->urls = new std::vector<Url*>;
    this->httpResponseCode = -1;
    this->baseUrlHttps = false;
    this->clientErrorText = "null";
    this->loadingTime = -1;
}

UrlCollection::~UrlCollection() {
    for(Url* x: *urls){
        delete x;
    }
    delete urls;
}

std::string UrlCollection::getBaseUrl() { return baseUrl;}

DataModulesEnum UrlCollection::getDataModuleType() {
    return URL_MODULE;
}

void UrlCollection::addUrl(Url* url) {
    urls->push_back(url);
}

std::vector<Url*>* UrlCollection::getUrls(){ return urls; }

bool UrlCollection::isHttps(){ return baseUrlHttps;}

int UrlCollection::getHttpResponseCode() const { return httpResponseCode;  }

void UrlCollection::setHttpResponseCode(int httpResponseCode){
    this->httpResponseCode = httpResponseCode;
}

void UrlCollection::setBaseUrl(const std::string &baseUrl) {
    UrlCollection::baseUrl = baseUrl;
}

void UrlCollection::setBaseUrlHttps(bool baseUrlHttps) {
    UrlCollection::baseUrlHttps = baseUrlHttps;
}

const std::string &UrlCollection::getClientErrorText() const {
    return clientErrorText;
}

void UrlCollection::setClientErrorText(const std::string &clientErrorText) {
    UrlCollection::clientErrorText = clientErrorText;
}

int UrlCollection::getLoadingTime() const {
    return loadingTime;
}

void UrlCollection::setLoadingTime(int loadingTime) {
    UrlCollection::loadingTime = loadingTime;
}

size_t UrlCollection::getSize() const {
    return size;
}

void UrlCollection::setSize(size_t size) {
    UrlCollection::size = size;
}


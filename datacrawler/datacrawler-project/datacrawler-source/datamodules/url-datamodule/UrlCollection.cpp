//
// Created by doktorgibson on 1/19/19.
//

#include "UrlCollection.h"

UrlCollection::UrlCollection() {
    this->urls = new std::list<Url>;
    this->httpResponseCode = -1;
    this->baseUrlHttps = false;
    this->clientErrorText = "null";
    this->loadingTime = -1;
}

DataModulesEnum UrlCollection::getDataModules()  {
    return URL_MODULE;
}


UrlCollection::~UrlCollection() {
    delete urls;
}

std::string UrlCollection::getBaseUrl() { return baseUrl;}

void UrlCollection::addUrl(std::string urlText, std::string url) {
    Url tmpUrl(urlText, url);
    urls->push_back(tmpUrl);
}

std::list<Url>* UrlCollection::getUrls(){ return urls; }

bool UrlCollection::isHttps(){ return baseUrlHttps;}

int UrlCollection::getHttpResponseCode() { return httpResponseCode;  }

void UrlCollection::setHttpResponseCode(int httpResponseCode){
    this->httpResponseCode = httpResponseCode;
}

void UrlCollection::setBaseUrl(std::string baseUrl) {
    UrlCollection::baseUrl = baseUrl;
}

void UrlCollection::setBaseUrlHttps(bool baseUrlHttps) {
    UrlCollection::baseUrlHttps = baseUrlHttps;
}

const std::string UrlCollection::getClientErrorText()  {
    return clientErrorText;
}

void UrlCollection::setClientErrorText(std::string clientErrorText) {
    UrlCollection::clientErrorText = clientErrorText;
}

int UrlCollection::getLoadingTime() {
    return loadingTime;
}

void UrlCollection::setLoadingTime(int loadingTime) {
    UrlCollection::loadingTime = loadingTime;
}

size_t UrlCollection::getSize() {
    return size;
}

void UrlCollection::setSize(size_t size) {
    UrlCollection::size = size;
}

std::string UrlCollection::getTitle() {
    return title;
}

void UrlCollection::setTitle(std::string title) {
    UrlCollection::title = title;
}


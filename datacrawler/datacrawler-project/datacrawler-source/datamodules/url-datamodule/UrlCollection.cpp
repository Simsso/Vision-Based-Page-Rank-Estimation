//
// Created by doktorgibson on 1/19/19.
//

#include "UrlCollection.h"

UrlCollection::UrlCollection(std::string baseUrl) {
    this->baseUrl = baseUrl;
    this->urls = new std::vector<Url*>;
}

UrlCollection::~UrlCollection() {
    for(auto x: *urls){
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
//
// Created by doktorgibson on 1/13/19.
//

#include "Url.h"

Url::Url(std::string urlText, std::string url, bool isHttps){
    this->urlText = urlText;
    this->url = url;
    this->https = isHttps;
}

Url::~Url(){}

std::string Url::getUrlText() { return urlText;}

std::string Url::getUrl(){ return url;}

bool Url::isHttps() { return https;}
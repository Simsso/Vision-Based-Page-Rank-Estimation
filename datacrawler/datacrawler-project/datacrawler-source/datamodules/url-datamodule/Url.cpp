//
// Created by doktorgibson on 1/13/19.
//

#include "Url.h"

Url::Url(std::string urlText, std::string url){
    this->urlText = urlText;
    this->url = url;
}

Url::~Url(){}

Url::Url(const Url& url){
    this->url = url.url;
    this->urlText = url.urlText;
}

std::string Url::getUrlText() { return urlText;}

std::string Url::getUrl(){ return url;}

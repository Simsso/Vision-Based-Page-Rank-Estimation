//
// Created by doktorgibson on 1/13/19.
//

#include "Url.h"

Url::Url(std::string urlText, std::string url){
    this->urlText = urlText;
    this->url = url;
}

Url::~Url(){}

std::string Url::getUrlText() { return urlText;}

std::string Url::getUrl(){ return url;}

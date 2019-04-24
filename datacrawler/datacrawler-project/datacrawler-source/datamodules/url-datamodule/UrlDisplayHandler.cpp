//
// Created by doktorgibson on 1/30/19.
//

#include "UrlDisplayHandler.h"

UrlDisplayHandler::UrlDisplayHandler(std::string* title) {
    logger = Logger::getInstance();
    this->title = title;
    this->wasCalled = false;
}

void UrlDisplayHandler::OnTitleChange( CefRefPtr< CefBrowser > browser, const CefString& title ){

    if(!wasCalled){
        *this->title = title.ToString();
        wasCalled = true;
    }
}
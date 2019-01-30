//
// Created by doktorgibson on 1/30/19.
//

#include "UrlDisplayHandler.h"

UrlDisplayHandler::UrlDisplayHandler(std::string& title) : title(title) {
}

void UrlDisplayHandler::OnTitleChange( CefRefPtr< CefBrowser > browser, const CefString& title ){
    this->title = title.ToString();
}
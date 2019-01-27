//
// Created by doktorgibson on 1/27/19.
//

#include "UrlRequestHandler.h"

UrlRequestHandler::UrlRequestHandler(UrlResponseFilter* urlResponseFilter) {
    this->urlResponseFilter = urlResponseFilter;
    this->logger = Logger::getInstance();
}

UrlRequestHandler::~UrlRequestHandler() {}

CefRefPtr<CefResponseFilter> UrlRequestHandler::GetResourceResponseFilter(
        CefRefPtr<CefBrowser> browser,
        CefRefPtr<CefFrame> frame,
        CefRefPtr<CefRequest> request,
        CefRefPtr<CefResponse> response) {
    return urlResponseFilter;
}

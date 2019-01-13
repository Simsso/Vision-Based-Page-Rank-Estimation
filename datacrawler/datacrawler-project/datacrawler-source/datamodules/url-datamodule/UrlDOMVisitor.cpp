//
// Created by doktorgibson on 1/13/19.
//

#include "UrlDOMVisitor.h"

UrlDOMVisitor::UrlDOMVisitor() {
    logger = Logger::getInstance();
}

UrlDOMVisitor::~UrlDOMVisitor() {}

void UrlDOMVisitor::Visit(CefRefPtr<CefDOMDocument> domDocument){
    logger->info("Parsing URLs !");
}

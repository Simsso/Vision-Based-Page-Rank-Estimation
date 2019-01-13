//
// Created by doktorgibson on 1/13/19.
//
#include "UrlLoadHandler.h"

UrlLoadHandler::UrlLoadHandler() {
    logger = Logger::getInstance();
}

UrlLoadHandler::~UrlLoadHandler() {}

void UrlLoadHandler::OnLoadEnd(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, int httpStatusCode){
    logger->info("DOM was loaded!");

    logger->info("Notifiying render process for URL crawl!");
    CefRefPtr<CefProcessMessage> processMessage;
    browser.get()->SendProcessMessage(PID_RENDERER, processMessage.get()->Create("GetAllUrl"));
}
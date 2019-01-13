//
// Created by doktorgibson on 1/13/19.
//

#include "UrlClient.h"

UrlClient::UrlClient() {
    logger = Logger::getInstance();
}

UrlClient::~UrlClient() {}

UrlClient::UrlClient(UrlLoadHandler* urlLoadHandler, UrlRenderHandler* urlRenderHandler) {
    logger = Logger::getInstance();
    this->urlLoadHandler = urlLoadHandler;
    this->urlRenderHandler = urlRenderHandler;
}

CefRefPtr<CefLoadHandler> UrlClient::GetLoadHandler() {
    return urlLoadHandler;
}

CefRefPtr<CefRenderHandler> UrlClient::GetRenderHandler(){
    return urlRenderHandler;
}


bool UrlClient::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
                                                               CefProcessId source_process,
                                                               CefRefPtr<CefProcessMessage> message) {

    if(message.get()->GetName() == "GetAllUrl_finished") {
        logger->info("URL-Datamdoudle received event from RenderProcessHandler! Parsing finished!");
    }
    return false;
}
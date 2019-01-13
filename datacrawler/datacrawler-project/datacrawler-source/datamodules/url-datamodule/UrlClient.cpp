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
        logger->info("URL-Datamodule received event from RenderProcessHandler! Parsing finished!");
        CefRefPtr<CefListValue> listValue = message.get()->GetArgumentList();
        logger->info(listValue->GetString(0));

        logger->info("Running URL-Datamodule .. finished !");
        CefQuitMessageLoop();
    }
    return false;
}
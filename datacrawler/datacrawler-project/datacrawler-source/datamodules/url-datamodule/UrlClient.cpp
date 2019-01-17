//
// Created by doktorgibson on 1/13/19.
//

#include "UrlClient.h"

UrlClient::UrlClient() {
    logger = Logger::getInstance();
}

UrlClient::~UrlClient() {}

UrlClient::UrlClient(UrlLoadHandler* urlLoadHandler, UrlRenderHandler* urlRenderHandler, vector<Url*>* urls) {
    logger = Logger::getInstance();
    this->urlLoadHandler = urlLoadHandler;
    this->urlRenderHandler = urlRenderHandler;
    this->urls = urls;
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
        CefRefPtr <CefListValue> listUrls = listValue.get()->GetList(0);
        CefRefPtr <CefListValue> listText = listValue.get()->GetList(1);

        logger->info("Constructing Url-objects .. !");
        regex httpsRegex("^https://");
        bool calculatedUrlHasHttps;

        for(size_t i=0; i < listUrls.get()->GetSize(); i++){
            string tmpUrl = listUrls.get()->GetString(i);
            string tmpText = listText.get()->GetString(i);

            if(regex_search(tmpUrl, httpsRegex))
                calculatedUrlHasHttps = true;
            else
                calculatedUrlHasHttps = false;

            Url * tmp = new Url(tmpText, tmpUrl, calculatedUrlHasHttps);
            urls->push_back(tmp);
        }

        logger->info("Running URL-Datamodule .. finished !");
        CefQuitMessageLoop();
    }
    return false;
}
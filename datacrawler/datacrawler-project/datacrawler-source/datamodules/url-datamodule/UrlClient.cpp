//
// Created by doktorgibson on 1/13/19.
//

#include "UrlClient.h"

UrlClient::UrlClient() {
    logger = Logger::getInstance();
}

UrlClient::~UrlClient() {}

UrlClient::UrlClient(UrlLoadHandler* urlLoadHandler, UrlRenderHandler* urlRenderHandler, UrlCollection* urlCollection, UrlRequestHandler* urlRequestHandler) {
    logger = Logger::getInstance();
    this->urlLoadHandler = urlLoadHandler;
    this->urlRenderHandler = urlRenderHandler;
    this->urlCollection = urlCollection;
    this->urlRequestHandler = urlRequestHandler;
}

CefRefPtr<CefLoadHandler> UrlClient::GetLoadHandler() {
    return urlLoadHandler;
}

CefRefPtr<CefRenderHandler> UrlClient::GetRenderHandler(){
    return urlRenderHandler;
}

CefRefPtr<CefRequestHandler> UrlClient::GetRequestHandler(){
    return urlRequestHandler;
}

bool UrlClient::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
                                                               CefProcessId source_process,
                                                               CefRefPtr<CefProcessMessage> message) {

    if(message.get()->GetName() == "GetAllUrl_finished") {
        logger->info("URL-Datamodule received event from RenderProcessHandler! Parsing finished!");
        string baseUrl = browser.get()->GetMainFrame().get()->GetURL();

        CefRefPtr<CefListValue> listValue = message.get()->GetArgumentList();
        CefRefPtr <CefListValue> listUrls = listValue.get()->GetList(0);
        CefRefPtr <CefListValue> listText = listValue.get()->GetList(1);

        logger->info("Constructing Url-objects .. !");
        regex httpsRegex("^https://");

        if(regex_search(baseUrl, httpsRegex))
            urlCollection->setBaseUrlHttps(true);
        else
            urlCollection->setBaseUrlHttps(false);

        urlCollection->setBaseUrl(baseUrl);

        urlCollection->setHttpResponseCode(listValue.get()->GetInt(2));

        urlCollection->setLoadingTime(listValue.get()->GetInt(3));

        for(size_t i=0; i < listUrls.get()->GetSize(); i++){
            string tmpUrl = listUrls.get()->GetString(i);
            string tmpText = listText.get()->GetString(i);

            Url * tmp = new Url(tmpText, tmpUrl);
            urlCollection->addUrl(tmp);
        }
        logger->info("Running URL-Datamodule .. finished !");

        CefQuitMessageLoop();
    } else if(message.get()->GetName() == "LoadingFailed") {
        string clientErrorText = message->GetArgumentList()->GetString(0);
        string baseUrl = message->GetArgumentList()->GetString(1);
        urlCollection->setClientErrorText(clientErrorText);
        urlCollection->setBaseUrl(baseUrl);
        CefQuitMessageLoop();
    }

    return false;
}
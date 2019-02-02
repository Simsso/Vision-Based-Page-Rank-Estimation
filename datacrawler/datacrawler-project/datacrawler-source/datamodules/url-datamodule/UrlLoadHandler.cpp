//
// Created by doktorgibson on 1/13/19.
//
#include "UrlLoadHandler.h"

UrlLoadHandler::UrlLoadHandler() {
    logger = Logger::getInstance();
}

UrlLoadHandler::UrlLoadHandler(string url) {
    logger = Logger::getInstance();
    this->url = url;
    notified = false;
    failed = false;
}

UrlLoadHandler::~UrlLoadHandler() {
}

void UrlLoadHandler::OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                 CefLoadHandler::ErrorCode errorCode, const CefString &errorText,
                                 const CefString &failedUrl) {

    if(errorCode == -27 || errorCode == -3) // Skip ERROR_BLOCKED_BY_RESPONSE, ERR_ABORTED
        return;

    failed = true;
    logger->info("Failed to load!");
    logger->info(errorText.ToString()+" "+to_string(errorCode));
    CefRefPtr<CefProcessMessage> processMessage = CefProcessMessage::Create("LoadingFailed");
    processMessage.get()->GetArgumentList().get()->SetString(0, errorText);
    processMessage.get()->GetArgumentList().get()->SetString(1, failedUrl);
    browser.get()->SendProcessMessage(PID_RENDERER, processMessage);
}

void UrlLoadHandler::OnLoadStart( CefRefPtr< CefBrowser > browser, CefRefPtr< CefFrame > frame, CefLoadHandler::TransitionType transition_type ){
}

void UrlLoadHandler::OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                                  bool isLoading,
                                  bool canGoBack,
                                  bool canGoForward) {

    if(isLoading && !stoppedLoadingStartTime){
        loadingStartTime = std::chrono::steady_clock::now();
        stoppedLoadingStartTime = true;
    }

    if(!isLoading && !notified && !failed){
        logger->info("DOM was loaded!");

        logger->info("Notifiying render process for URL crawl!");

        CefRefPtr<CefProcessMessage> processMessage = CefProcessMessage::Create("GetAllUrl");
        processMessage.get()->GetArgumentList().get()->SetString(0, url);
        processMessage.get()->GetArgumentList().get()->SetInt(1, httpStatusCode);

        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        int loadingTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - loadingStartTime).count();
        processMessage.get()->GetArgumentList().get()->SetInt(2, loadingTime);

        logger->info("Loaded in "+to_string(loadingTime)+"ms");

        browser.get()->SendProcessMessage(PID_RENDERER, processMessage);

        notified = true;
    }
}

void UrlLoadHandler::OnLoadEnd(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, int httpStatusCode){
   if(frame->IsMain()){
       this->httpStatusCode = httpStatusCode;
   }
}
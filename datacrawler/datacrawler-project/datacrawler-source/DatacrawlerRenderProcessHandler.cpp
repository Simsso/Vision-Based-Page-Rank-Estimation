//
// Created by doktorgibson on 1/13/19.
//

#include "DatacrawlerRenderProcessHandler.h"

DatacrawlerRenderProcessHandler::DatacrawlerRenderProcessHandler(){
    logger = Logger::getInstance();
    urlDOMVisitor = new UrlDOMVisitor();
}

DatacrawlerRenderProcessHandler::~DatacrawlerRenderProcessHandler(){}

bool DatacrawlerRenderProcessHandler::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
                                      CefProcessId source_process,
                                      CefRefPtr<CefProcessMessage> message) {

    if(message.get()->GetName() == "GetAllUrl") {
        logger->info("RenderProcessHandler received event from URL-Datamodule!");

        CefRefPtr<CefFrame> mainFrame = browser.get()->GetMainFrame();
        mainFrame.get()->VisitDOM(urlDOMVisitor);

        // send message to browser of URL-Datamodule, that we are finished
        CefRefPtr<CefProcessMessage> processMessage;
        browser.get()->SendProcessMessage(PID_BROWSER, processMessage.get()->Create("GetAllUrl_finished"));
    }
    return false;
}
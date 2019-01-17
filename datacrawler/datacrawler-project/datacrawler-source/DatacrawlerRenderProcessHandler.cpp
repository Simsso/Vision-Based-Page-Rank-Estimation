//
// Created by doktorgibson on 1/13/19.
//

#include "DatacrawlerRenderProcessHandler.h"

DatacrawlerRenderProcessHandler::DatacrawlerRenderProcessHandler(){
    logger = Logger::getInstance();

}

DatacrawlerRenderProcessHandler::~DatacrawlerRenderProcessHandler(){}

bool DatacrawlerRenderProcessHandler::OnProcessMessageReceived(CefRefPtr<CefBrowser> browser,
                                      CefProcessId source_process,
                                      CefRefPtr<CefProcessMessage> message) {
    /* URL-Datamodule*/
    if(message.get()->GetName() == "GetAllUrl") {
        CefRefPtr<UrlDOMVisitor> urlDOMVisitor(new UrlDOMVisitor(urls, message.get()->GetArgumentList().get()->GetString(0), 10));
        logger->info("RenderProcessHandler received event from URL-Datamodule!");

        CefRefPtr<CefFrame> mainFrame = browser.get()->GetMainFrame();
        // delegate parsing to UrlDomVisitor
        mainFrame.get()->VisitDOM(urlDOMVisitor);

        // send message to browser of URL-Datamodule, that we are finished
        CefRefPtr<CefProcessMessage> processMessage = CefProcessMessage::Create("GetAllUrl_finished");
        CefRefPtr <CefListValue> listUrls = CefListValue::Create();
        CefRefPtr <CefListValue> listUrlsText = CefListValue::Create();


        for(unsigned long i = 0; i < urls.size(); i++){
             listUrls.get()->SetString(i, urls.at(i).first);
             listUrlsText.get()->SetString(i, urls.at(i).second);
        }

        processMessage.get()->GetArgumentList()->SetList(0, listUrls);
        processMessage.get()->GetArgumentList()->SetList(1, listUrlsText);
        browser.get()->SendProcessMessage(PID_BROWSER, processMessage);
    }
    return false;
}
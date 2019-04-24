//
// Created by doktorgibson on 1/10/19.
//

#include "ScreenshotRequestHandler.h"


/**
 * ScreenshotRequestHandler
 */
ScreenshotRequestHandler::ScreenshotRequestHandler(){}

/**
 * ~ScreenshotRequestHandler
 */
ScreenshotRequestHandler::~ScreenshotRequestHandler(){}

ScreenshotRequestHandler::ScreenshotRequestHandler(std::map<std::string, std::string> headerMap){
    this->headerMap = headerMap;
}

/**
 *
 */
CefRequestHandler::ReturnValue ScreenshotRequestHandler::OnBeforeResourceLoad(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                                                      CefRefPtr<CefRequest> request, CefRefPtr<CefRequestCallback> callback) {
    CefRequest::HeaderMap requestHeaderMap;
    request->GetHeaderMap(requestHeaderMap);

    for( auto const& x : headerMap ) {
        requestHeaderMap.insert(std::make_pair(x.first, x.second));
    }

    request->SetHeaderMap(requestHeaderMap);
    return RV_CONTINUE;
}
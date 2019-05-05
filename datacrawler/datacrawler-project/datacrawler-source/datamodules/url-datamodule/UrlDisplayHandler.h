//
// Created by doktorgibson on 1/30/19.
//

#ifndef DATACRAWLER_PROJECT_URLDISPLAYHANDLER_H
#define DATACRAWLER_PROJECT_URLDISPLAYHANDLER_H

#include <include/cef_display_handler.h>
#include "../../util/Logger.h"

class UrlDisplayHandler : public CefDisplayHandler {
    IMPLEMENT_REFCOUNTING(UrlDisplayHandler)
private:
    std::string * title;
    Logger * logger;
    bool wasCalled;

public:
     void OnTitleChange( CefRefPtr< CefBrowser >, const CefString&) OVERRIDE;

     UrlDisplayHandler(std::string*);
};


#endif //DATACRAWLER_PROJECT_URLDISPLAYHANDLER_H

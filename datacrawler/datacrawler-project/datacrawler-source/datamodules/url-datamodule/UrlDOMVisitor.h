//
// Created by doktorgibson on 1/13/19.
//

#ifndef DATACRAWLER_PROJECT_URLDOMVISITOR_H
#define DATACRAWLER_PROJECT_URLDOMVISITOR_H


#include <include/cef_dom.h>
#include "../../util/Logger.h"

class UrlDOMVisitor: public CefDOMVisitor {
private:
    IMPLEMENT_REFCOUNTING(UrlDOMVisitor)
    Logger* logger;

public:
    void Visit(CefRefPtr<CefDOMDocument>) OVERRIDE;

    UrlDOMVisitor();
    ~UrlDOMVisitor();
};


#endif //DATACRAWLER_PROJECT_URLDOMVISITOR_H

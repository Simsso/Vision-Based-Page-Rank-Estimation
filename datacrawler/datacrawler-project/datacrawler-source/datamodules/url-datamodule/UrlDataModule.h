//
// Created by doktorgibson on 1/12/19.
//

#ifndef DATACRAWLER_PROJECT_URLDATAMODULE_H
#define DATACRAWLER_PROJECT_URLDATAMODULE_H

#include <include/internal/cef_ptr.h>
#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>

#include "../DataModuleBase.h"
#include "UrlClient.h"
#include "UrlCollection.h"


class UrlDataModule: public DataModuleBase, public CefBaseRefCounted {
private:
    IMPLEMENT_REFCOUNTING(UrlDataModule);
    vector<Url*> *urls;
    string * baseUrl;

public:
    DataBase* process(std::string) OVERRIDE;

    UrlDataModule();
    ~UrlDataModule();
};


#endif //DATACRAWLER_PROJECT_URLDATAMODULE_H

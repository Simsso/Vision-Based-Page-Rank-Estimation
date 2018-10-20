//
// Created by samed on 18.10.18.
//

#ifndef EXPERIMENT_PROJECT_EXPERIMENT_APP_H
#define EXPERIMENT_PROJECT_EXPERIMENT_APP_H


#include <include/cef_app.h>
#include "experiment_handler.h"

class ExperimentClient: public CefClient {
private:
    CefRefPtr<CefRenderHandler> experimentHandler;

public:
    ExperimentClient(ExperimentHandler* handler){
        experimentHandler = handler;
    }

    // Override CefClient methods
    CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE {
        return experimentHandler;
    }


private:
    IMPLEMENT_REFCOUNTING(ExperimentClient)
};


#endif //EXPERIMENT_PROJECT_EXPERIMENT_APP_H

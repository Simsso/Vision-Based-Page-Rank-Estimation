//
// Created by Samed Güner on 18.10.18.
//

#include <include/internal/cef_linux.h>
#include "experiment_client.h"
#include "iostream"

using namespace std;


int main(int argc, char* argv[]){

    CefMainArgs mainArgs(argc, argv);
    CefExecuteProcess(mainArgs, NULL, NULL);

    CefSettings cefSettings;

    // Initialize CEF for the browser process
    CefInitialize(mainArgs, cefSettings, NULL, NULL);

    ExperimentHandler* experimentHandler = new ExperimentHandler(1920, 1080);

    // CefRefPtr represents a SmartPointer (Releases Object once function returns)
    //
    // ExperimentClient implements application-level callbacks for the browser process
    // pass the custom CefRenderHandler object
    CefRefPtr<ExperimentClient> client = new ExperimentClient(experimentHandler);

    CefWindowInfo cefWindowInfo;
    cefWindowInfo.SetAsWindowless(0);

    CefBrowserSettings browserSettings;

    // Create a synchronous browser, will be created once CEF has initialized
    CefRefPtr<CefBrowser> browser;
    browser = CefBrowserHost::CreateBrowserSync(cefWindowInfo, client.get(), "https://timodenk.com", browserSettings, NULL);

    CefRunMessageLoop();

    CefShutdown();
}

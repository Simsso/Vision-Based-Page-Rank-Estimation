#ifndef DATACRAWLER_PROJECT_DATACRAWLER_H
#define DATACRAWLER_PROJECT_DATACRAWLER_H

#include <iostream>
#include <list>

#include <include/internal/cef_linux.h>
#include <include/cef_app.h>
#include <include/wrapper/cef_helpers.h>

#include "DatacrawlerConfiguration.h"
#include "datamodules/DataModuleBase.h"
#include "util/Logger.h"


using namespace std;

class Datacrawler {
private:
   DatacrawlerConfiguration datacrawlerConfiguration;
   int numNodes;
   list<DataModuleBase*> dataModules;
   Logger* logger;
   map<std::string, NodeElement*>* graph;

   vector<pair<string, NodeElement*>>  buildNodes(NodeElement*);

public:
    map<string,NodeElement*> * process(string);
    void init();

    Datacrawler();
    ~Datacrawler();
};


#endif //DATACRAWLER_PROJECT_DATACRAWLER_H

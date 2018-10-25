//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_DATAMODULEBASE_H
#define DATACRAWLER_PROJECT_DATAMODULEBASE_H

#include "../util/Logger.h"
#include "../graph/NodeElement.h"

class DataModuleBase {

protected:
    Logger* logger;
public:
    virtual NodeElement* process();

    DataModuleBase();
    ~DataModuleBase();
};


#endif //DATACRAWLER_PROJECT_DATAMODULEBASE_H

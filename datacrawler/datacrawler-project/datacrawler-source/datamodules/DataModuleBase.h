#ifndef DATACRAWLER_PROJECT_DATAMODULEBASE_H
#define DATACRAWLER_PROJECT_DATAMODULEBASE_H

#include "../util/Logger.h"
#include "../graph/NodeElement.h"

class DataModuleBase {

protected:
    Logger* logger;
    std::string url;

public:
    virtual DataBase* process(std::string);

    DataModuleBase();
    ~DataModuleBase();
};


#endif //DATACRAWLER_PROJECT_DATAMODULEBASE_H

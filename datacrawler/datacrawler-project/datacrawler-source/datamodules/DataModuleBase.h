#ifndef DATACRAWLER_PROJECT_DATAMODULEBASE_H
#define DATACRAWLER_PROJECT_DATAMODULEBASE_H

#include <include/internal/cef_linux.h>
#include "../util/Logger.h"
#include "../graph/NodeElement.h"

class DataModuleBase {

protected:
    Logger* logger;

public:
    virtual DataBase* process(CefMainArgs*, std::string);

    DataModuleBase();
    ~DataModuleBase();
};


#endif //DATACRAWLER_PROJECT_DATAMODULEBASE_H

#ifndef DATACRAWLER_PROJECT_DATAMODULEBASE_H
#define DATACRAWLER_PROJECT_DATAMODULEBASE_H

#include <include/internal/cef_linux.h>
#include "../util/Logger.h"
#include "../graph/NodeElement.h"

class DataModuleBase {

protected:
    Logger* logger;

public:
    virtual DataBase* process(std::string);

    DataModuleBase();
    virtual ~DataModuleBase();
};


#endif //DATACRAWLER_PROJECT_DATAMODULEBASE_H

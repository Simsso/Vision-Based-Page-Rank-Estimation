#ifndef DATACRAWLER_PROJECT_DATABASE_H
#define DATACRAWLER_PROJECT_DATABASE_H


#include "../DataModulesEnum.h"
#include "url-datamodule/Url.h"
#include <vector>

class DataBase {
protected:
    DataModulesEnum dataModulesEnum;

public:
    virtual DataModulesEnum getDataModules() = 0;
    virtual ~DataBase();
};


#endif //DATACRAWLER_PROJECT_DATABASE_H

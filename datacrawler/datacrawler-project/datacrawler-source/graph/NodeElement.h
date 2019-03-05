#ifndef DATACRAWLER_PROJECT_NODEELEMENT_H
#define DATACRAWLER_PROJECT_NODEELEMENT_H

#include "../datamodules/DataBase.h"
#include <vector>

class NodeElement {

private:
   std::vector<DataBase*>* data;

public:
    void addData(DataBase*);
    std::vector<DataBase*> * getData();

    NodeElement();
    ~NodeElement();
};


#endif //DATACRAWLER_PROJECT_NODEELEMENT_H

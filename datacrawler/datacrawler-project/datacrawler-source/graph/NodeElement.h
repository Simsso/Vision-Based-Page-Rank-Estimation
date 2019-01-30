#ifndef DATACRAWLER_PROJECT_NODEELEMENT_H
#define DATACRAWLER_PROJECT_NODEELEMENT_H

#include <list>

#include "../datamodules/DataBase.h"

class NodeElement {

private:
   std::list<DataBase*> data;
   bool startNode;

public:
    void addData(DataBase*);
    std::list<DataBase*> getData();
    bool isStartNode();

    NodeElement(bool);
    ~NodeElement();
};


#endif //DATACRAWLER_PROJECT_NODEELEMENT_H

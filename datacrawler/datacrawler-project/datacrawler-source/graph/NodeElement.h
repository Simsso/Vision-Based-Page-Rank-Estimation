//
// Created by samed on 23.10.18.
//

#ifndef DATACRAWLER_PROJECT_NODEELEMENT_H
#define DATACRAWLER_PROJECT_NODEELEMENT_H

#include "iostream"

using namespace std;

class NodeElement {

private:
    string type;

public:

    string getType();

    NodeElement();
    NodeElement(string type);
    ~NodeElement();
};


#endif //DATACRAWLER_PROJECT_NODEELEMENT_H

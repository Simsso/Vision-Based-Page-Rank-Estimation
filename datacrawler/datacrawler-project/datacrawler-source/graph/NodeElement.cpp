#include "NodeElement.h"

/**
 * ~NodelElement
 */
NodeElement::~NodeElement() {
    for(DataBase* database: data){
        delete database;
    }
}

/**
 * NodeElement
 */
NodeElement::NodeElement(bool isStartNode){
    startNode = isStartNode;
}

/**
 * addData - Adds data to this NodeElement
 * @param newData, which shall be added
 */
void NodeElement::addData(DataBase * newData) {
    data.push_back(newData);
}

/**
 * getData - Returns the DataBase-instances saved in this NodeElement
 * @return A pointer on to a vector<DataBase*> having all DataBase-instances of this NodeElement
 */
std::list<DataBase*> NodeElement::getData(){ return data;}

bool NodeElement::isStartNode(){
    return startNode;
}
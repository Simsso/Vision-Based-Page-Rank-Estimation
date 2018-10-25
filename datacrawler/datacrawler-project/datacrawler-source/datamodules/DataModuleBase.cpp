//
// Created by samed on 23.10.18.
//

#include "DataModuleBase.h"

DataModuleBase::~DataModuleBase(){}
DataModuleBase::DataModuleBase(){
    logger = Logger::getInstance();
}

NodeElement* DataModuleBase::process(){
    return nullptr ;
}
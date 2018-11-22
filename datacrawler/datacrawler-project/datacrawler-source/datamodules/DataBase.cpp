//
// Created by doktorgibson on 11/22/18.
//

#include "DataBase.h"

/**
 * ~DataBase
 */
DataBase::~DataBase() {}

/**
 * getDataModuleType - Virtual function, which has to be overwritten by derivates of this class.
 * @return Type of the DataModule, which owns this DataBase
 */
DataModulesEnum DataBase::getDataModuleType() {
    return NO_MODULE;
}
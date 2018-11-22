#include "DataModuleBase.h"

/**
 * ~DataModuleBase
 */
DataModuleBase::~DataModuleBase(){}

/**
 * DataModuleBase
 */
DataModuleBase::DataModuleBase(){
    logger = Logger::getInstance();
}

/**
 * process - Virtual function, which has to be overwritten by derivates of this class (DataModule).
 * @param url, which shall be processed by the given DataModule
 * @return nullptr if not implemented, otherwise DataBase of the specific DataModule instance
 */
DataBase* DataModuleBase::process(std::string url){
    this->url = url;
    return nullptr;
}
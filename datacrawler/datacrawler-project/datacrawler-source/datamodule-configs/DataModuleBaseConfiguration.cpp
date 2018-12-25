#include "DataModuleBaseConfiguration.h"

/**
 * ~DataModuleBaseConfiguration
 */
DataModuleBaseConfiguration::~DataModuleBaseConfiguration(){}

/**
 * DataModuleBaseConfiguration
 */
DataModuleBaseConfiguration::DataModuleBaseConfiguration(){}

/**
 * createInstance - Virtual function, which has to be overwritten by the specific DataModule-derivates
 * @return Returns nullptr, if implemented returns instance of DataModule
 */
DataModuleBase* DataModuleBaseConfiguration::createInstance() {
    return nullptr;
}
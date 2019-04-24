#ifndef DATACRAWLER_PROJECT_DATAMODULESENUM_H
#define DATACRAWLER_PROJECT_DATAMODULESENUM_H

#include <string>

/**
 * DataModulesEnum - List of IDs representing the currently available DataModules in the crawler
 */
enum DataModulesEnum {
    NO_MODULE = 0,
    SCREENSHOT_MODULE = 1,
    SCREENSHOT_MOBILE_MODULE = 2,
    URL_MODULE = 3
};


inline std::string toStringDataModulesEnum(DataModulesEnum v)
{
    switch (v)
    {
        case SCREENSHOT_MODULE: return "SCREENSHOT_MODULE";
        case SCREENSHOT_MOBILE_MODULE: return "SCREENSHOT_MOBILE_MODULE";
        default: return "NO_MODULE";
    }
}

#endif //DATACRAWLER_PROJECT_DATAMODULESENUM_H

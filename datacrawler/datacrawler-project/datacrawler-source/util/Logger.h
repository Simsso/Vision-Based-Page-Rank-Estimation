//
// Created by samed on 24.10.18.
//

#ifndef DATACRAWLER_PROJECT_LOGGER_H
#define DATACRAWLER_PROJECT_LOGGER_H

#include <iostream>
#include <iomanip>
#include <ctime>
#include "LogLevel.h"

class Logger {
private:
    static Logger* instance;
    LogLevel logLevel;
    Logger();

public:
    void info(std::string);
    void warn(std::string);
    void fatal(std::string);
    void debug(std::string);
    void error(std::string);
    static Logger* getInstance();
};

#endif //DATACRAWLER_PROJECT_LOGGER_H

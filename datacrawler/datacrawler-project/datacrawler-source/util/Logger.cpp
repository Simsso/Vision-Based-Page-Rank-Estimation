#include "Logger.h"

Logger::Logger() {
    if (char *logLevelEnv = std::getenv("LOG_LEVEL")) {
        std::string logLevel(logLevelEnv);

        if (logLevel.compare("LOG_ALL") == 0) {
            this->logLevel = LOG_ALL;
        } else if (logLevel.compare("LOG_INFO") == 0) {
            this->logLevel = LOG_INFO;
        } else if (logLevel.compare("LOG_DEBUG") == 0) {
            this->logLevel = LOG_DEBUG;
        } else if (logLevel.compare("LOG_ERROR") == 0) {
            this->logLevel = LOG_ERROR;
        } else if (logLevel.compare("LOG_FATAL") == 0) {
            this->logLevel = LOG_FATAL;
        } else if (logLevel.compare("LOG_WARN") == 0) {
            this->logLevel = LOG_WARN;
        }else if (logLevel.compare("LOG_OFF") == 0) {
            this->logLevel = LOG_OFF;
        } else {
            this->logLevel = LOG_ALL;
            info("Your log level is unknown! Using LOG_ALL as default!");
            return;
        }

        std::string str("Your log level is ");
        str.append(logLevelEnv);
        info(str);
        return;
    } else {
        this->logLevel = LOG_ALL;
        info("No log level was set! Using LOG_ALL as default!");
        return;
    }
}

void Logger::info(std::string text) {

    if (logLevel >= LOG_INFO) {
        auto time = std::time(nullptr);
        auto localTime = *std::localtime(&time);

        std::cout << std::put_time(&localTime, "[%H:%M:%S] ");
        std::cout << "- INFO - ";
        std::cout << text << std::endl;
    }
}

void Logger::warn(std::string text) {

    if (logLevel >= LOG_WARN) {
        auto time = std::time(nullptr);
        auto localTime = *std::localtime(&time);

        std::cout << std::put_time(&localTime, "[%H:%M:%S] ");
        std::cout << "- WARN - ";
        std::cout << text << std::endl;
    }

}

void Logger::fatal(std::string text) {

    if (logLevel >= LOG_FATAL) {
        auto time = std::time(nullptr);
        auto localTime = *std::localtime(&time);

        std::cout << std::put_time(&localTime, "[%H:%M:%S] ");
        std::cout << "- FATAL - ";
        std::cout << text << std::endl;
    }

}

void Logger::debug(std::string text) {

    if (logLevel >= LOG_DEBUG) {
        auto time = std::time(nullptr);
        auto localTime = *std::localtime(&time);

        std::cout << std::put_time(&localTime, "[%H:%M:%S] ");
        std::cout << "- DEBUG - ";
        std::cout << text << std::endl;
    }

}

void Logger::error(std::string text) {

    if (logLevel >= LOG_ERROR) {
        auto time = std::time(nullptr);
        auto localTime = *std::localtime(&time);

        std::cout << std::put_time(&localTime, "[%H:%M:%S] ");
        std::cout << "- ERROR - ";
        std::cout << text << std::endl;
    }

}

Logger *Logger::getInstance() {
    if (instance == 0)
        instance = new Logger();

    return instance;
}

#include "DatacrawlerConfiguration.h"

/**
 * ~DataCrawlerConfiguration
 */
DatacrawlerConfiguration::~DatacrawlerConfiguration() {
    for (auto datacrawlerConfiguration: configurations) {
        delete datacrawlerConfiguration.second;
    }
}

/**
 * DatacrawlerConfiguration - Loads all user-defined configurations for the graph to be generated
 */
DatacrawlerConfiguration::DatacrawlerConfiguration() {
    logger = Logger::getInstance();

    ifstream file("datacrawler.config.json");

    if (file) {
        json config;
        file >> config;
        //configurations[SCREENSHOT_MODULE] = new ScreenshotConfiguration(400, 400, ONPAINT_TIMEOUT, ELAPSED_TIME_ONPAINT_TIMEOUT, CHANGE_THRESHOLD, LAST_SCREENSHOTS, true);
        if (!config["DATAMODULES"].is_null()) {

            for (auto &datamoduleEntry : config["DATAMODULES"].items()) {
                auto datamoduleConfig = datamoduleEntry.value();

                if (datamoduleConfig["DATAMODULE"] == "SCREENSHOT_MODULE") {
                    configurations[SCREENSHOT_MODULE] = generateScreenshotDatamoduleConfig(
                            datamoduleConfig["ATTRIBUTES"]);
                }
                // else if (datamoduleConfig["DATAMODULE"] == "SCREENSHOT_MOBILE_MODULE")
                //    configurations[SCREENSHOT_MODULE] = generateScreenshotDatamoduleConfig(datamoduleConfig["ATTRIBUTES"]);
            }

        } else {
            logger->error("Missing 'DATAMODULES' array in config!");
        }
    } else {
        logger->error("Config file could not be opened!");
    }
}

ScreenshotConfiguration *DatacrawlerConfiguration::generateScreenshotDatamoduleConfig(json &attributes) {
    bool parseError = false;

    if (attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].is_null() &&
        !attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].is_number()) {
        logger->error(
                "Missing/Wrong value for 'ELAPSED_TIME_ONPAINT_TIMEOUT' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (attributes["ONPAINT_TIMEOUT"].is_null() && !attributes["ONPAINT_TIMEOUT"].is_number()) {
        logger->error("Missing/Wrong value for 'ONPAINT_TIMEOUT' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (attributes["CHANGE_THRESHOLD"].is_null() && !attributes["CHANGE_THRESHOLD"].is_number()) {
        logger->error("Missing/Wrong value for 'CHANGE_THRESHOLD' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (attributes["LAST_SCREENSHOTS"].is_null() && !attributes["LAST_SCREENSHOTS"].is_number()) {
        logger->error("Missing/Wrong value for 'LAST_SCREENSHOTS' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (attributes["HEIGHT"].is_null() && !attributes["HEIGHT"].is_number()) {
        logger->error("Missing/Wrong value for 'HEIGHT' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (attributes["WIDTH"].is_null() && !attributes["WIDTH"].is_number()) {
        logger->error("Missing/Wrong value for 'WIDTH' in config for ScreenshotDatamodule!");
        parseError = true;
    }

    if (parseError) {
        logger->error("ScreenshotDatamodule has configuration errors! Excluded!");
        return nullptr;
    }

    int elapsedTimeOnPaintTimeout = attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].get<int>();
    int onPaintTimeout = attributes["ONPAINT_TIMEOUT"].get<int>();
    double changeThreshold = attributes["CHANGE_THRESHOLD"].get<double>();
    int lastScreenshots = attributes["LAST_SCREENSHOTS"].get<int>();
    int height = attributes["HEIGHT"].get<int>();
    int width = attributes["WIDTH"].get<int>();

    return new ScreenshotConfiguration(height, width, onPaintTimeout, elapsedTimeOnPaintTimeout, changeThreshold,
                                       lastScreenshots, false);
}
/**
 * getConfiguration
 * @param selectedConfig represents the name of the DataModule for whom configuration shall be loaded
 * @return DataModuleBaseConfiguration, which represents the base class of all configurations for DataModules
 */
DataModuleBaseConfiguration *DatacrawlerConfiguration::getConfiguration(DataModulesEnum selectedConfig) {
    try {
        return configurations.at(selectedConfig);
    } catch (std::out_of_range ex) {
        return nullptr;
    }
}
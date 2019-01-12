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
 * DatacrawlerConfiguration - Loads all user-defined configurations for the graph to be generated from a json-file
 */
DatacrawlerConfiguration::DatacrawlerConfiguration() {
    logger = Logger::getInstance();

    ifstream file("datacrawler.config.json");

    if (file) {
        json config;
        file >> config;

        if (!config["DATAMODULES"].is_null()) {

            for (auto &datamoduleEntry : config["DATAMODULES"].items()) {
                auto datamoduleConfig = datamoduleEntry.value();

                if (datamoduleConfig["DATAMODULE"] == "SCREENSHOT_MODULE") {
                    if (datamoduleConfig["DISABLED"].is_null() || datamoduleConfig["DISABLED"].get<bool>())
                        continue;

                    configurations[SCREENSHOT_MODULE] = generateScreenshotDatamoduleConfig(
                            datamoduleConfig["ATTRIBUTES"], false);

                } else if (datamoduleConfig["DATAMODULE"] == "SCREENSHOT_MOBILE_MODULE") {
                    if (datamoduleConfig["DISABLED"].is_null() || datamoduleConfig["DISABLED"].get<bool>())
                        continue;

                    configurations[SCREENSHOT_MOBILE_MODULE] = generateScreenshotDatamoduleConfig(
                            datamoduleConfig["ATTRIBUTES"], true);

                } else if (datamoduleConfig["DATAMODULE"] == "URL_MODULE"){
                    if (datamoduleConfig["DISABLED"].is_null() || datamoduleConfig["DISABLED"].get<bool>())
                        continue;

                    configurations[URL_MODULE] = generateUrlDataModuleConfig(datamoduleConfig["ATTRIBUTES"]);
                }
            }

        } else {
            logger->error("Missing 'DATAMODULES' array in config!");
        }
    } else {
        logger->error("Config file could not be opened!");
    }
}

/**
 * generateScreenshotDatamoduleConfig
 * @param attributes represents a key-value storage for json-attributes retrieved for the Datamodule
 * @param mobile switches user-agent in Screenshot-Datamodule to mobile device if true. Otherwise default user-agent.
 * @return ScreenshotConfiguration*, which represents the configuration of the ScreenshotDatamodule
 */
ScreenshotConfiguration *DatacrawlerConfiguration::generateScreenshotDatamoduleConfig(json &attributes, bool mobile) {
    bool parseError = false;

    if(mobile)
        logger->info("<ScreenshotMobile-Datamodule> configuration");
    else
        logger->info("<Screenshot-Datamodule> configuration");

    if (attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].is_null() &&
        !attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].is_number()) {
        logger->error(
                "Missing/Wrong value for 'ELAPSED_TIME_ONPAINT_TIMEOUT' in config!");
        parseError = true;
    }

    if (attributes["ONPAINT_TIMEOUT"].is_null() && !attributes["ONPAINT_TIMEOUT"].is_number()) {
        logger->error("Missing/Wrong value for 'ONPAINT_TIMEOUT' in config!");
        parseError = true;
    }

    if (attributes["CHANGE_THRESHOLD"].is_null() && !attributes["CHANGE_THRESHOLD"].is_number()) {
        logger->error("Missing/Wrong value for 'CHANGE_THRESHOLD' in config!");
        parseError = true;
    }

    if (attributes["LAST_SCREENSHOTS"].is_null() && !attributes["LAST_SCREENSHOTS"].is_number()) {
        logger->error("Missing/Wrong value for 'LAST_SCREENSHOTS' in config!");
        parseError = true;
    }

    if (attributes["HEIGHT"].is_null() && !attributes["HEIGHT"].is_number()) {
        logger->error("Missing/Wrong value for 'HEIGHT' in config!");
        parseError = true;
    }

    if (attributes["WIDTH"].is_null() && !attributes["WIDTH"].is_number()) {
        logger->error("Missing/Wrong value for 'WIDTH' in config!");
        parseError = true;
    }

    if (parseError) {
        logger->error("Datamodule has configuration errors! Excluded!");
        return nullptr;
    }

    int elapsedTimeOnPaintTimeout = attributes["ELAPSED_TIME_ONPAINT_TIMEOUT"].get<int>();
    int onPaintTimeout = attributes["ONPAINT_TIMEOUT"].get<int>();
    double changeThreshold = attributes["CHANGE_THRESHOLD"].get<double>();
    int lastScreenshots = attributes["LAST_SCREENSHOTS"].get<int>();
    int height = attributes["HEIGHT"].get<int>();
    int width = attributes["WIDTH"].get<int>();

    logger->info(".. loaded !");

    return new ScreenshotConfiguration(height, width, onPaintTimeout, elapsedTimeOnPaintTimeout, changeThreshold,
                                       lastScreenshots, mobile);
}

UrlConfiguration *DatacrawlerConfiguration::generateUrlDataModuleConfig(json &attributes) {
    logger->info("<URL-Datamodule> configuration");

    logger->info(".. loaded !");

    return new UrlConfiguration();
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
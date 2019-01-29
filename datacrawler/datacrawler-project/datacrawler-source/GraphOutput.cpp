//
// Created by doktorgibson on 1/26/19.
//

#include <sys/stat.h>
#include "GraphOutput.h"

GraphOutput::~GraphOutput() {}

GraphOutput::GraphOutput(std::map<std::string, NodeElement *> * graph, std::string folderName) {
    this->graph = graph;
    this->folderName = folderName;
    logger = Logger::getInstance();
}

void GraphOutput::generateGraph() {
    logger->info("Outputting graph!");
    std::string fullPath = outputPath+folderName+"/";
    std::string imgFolderPath = fullPath+"image/";

    mkdir(fullPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(imgFolderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    logger->info("Created output folder: "+outputPath+folderName);

    nlohmann::json nodes;

    int nodeNum = 1;
    for(auto node : *graph){
        auto nodeData = node.second->getData();

        nlohmann::json nodeJson;
        nodeJson["id"] = nodeNum;
        nodeJson["startNode"] = node.second->isStartNode();

        for(auto entry : *nodeData){
            if(entry->getDataModuleType() == SCREENSHOT_MODULE || entry->getDataModuleType() == SCREENSHOT_MOBILE_MODULE){
                auto screenshot = (Screenshot*) entry;

                try {
                    std::vector<int> params;
                    params.push_back(CV_IMWRITE_JPEG_QUALITY);
                    params.push_back(70);

                    cv::Mat newImg = cv::Mat(screenshot->getHeight(), screenshot->getWidth(), CV_8UC4, screenshot->getScreenshot());
                    if(entry->getDataModuleType() == SCREENSHOT_MODULE)
                        cv::resize(newImg, newImg,cv::Size(screenshot->getWidth()/4, screenshot->getHeight()/4), 0, 0, CV_INTER_LINEAR);
                    else
                        cv::resize(newImg, newImg,cv::Size(screenshot->getWidth()/2, screenshot->getHeight()/2), 0, 0, CV_INTER_LINEAR);


                    std::string imgPath;
                    std::string fileName;

                    if(entry->getDataModuleType() == SCREENSHOT_MOBILE_MODULE) {
                        fileName = std::to_string(nodeNum) + "_mobile.jpg";
                        nodeJson["mobile_screenshot_filename"] = fileName;
                    } else {
                       fileName = std::to_string(nodeNum)+".jpg";
                       nodeJson["screenshot_filename"] = fileName;
                    }

                    imgPath.append(imgFolderPath);
                    imgPath.append(fileName);

                    cv::imwrite(imgPath, newImg, params);

                } catch(std::runtime_error& ex) {
                    fprintf(stderr, "Exception while converting taking picture of the website in JPG format: %s", ex.what());
                }
            } else if (entry->getDataModuleType() == URL_MODULE){
                UrlCollection* urlCollection = (UrlCollection*) entry;

                nodeJson["base_url"] = urlCollection->getBaseUrl();
                nodeJson["https"] = urlCollection->isHttps();

                int responseCode = urlCollection->getHttpResponseCode();
                std::string clientErrorText = urlCollection->getClientErrorText();

                int loadingTime = urlCollection->getLoadingTime();

                if (loadingTime == -1)
                    nodeJson["loading_time"] = nullptr;
                else
                    nodeJson["loading_time"] = loadingTime;

                if (responseCode == -1 )
                    nodeJson["server_status"] = nullptr;
                else
                    nodeJson["server_status"] = responseCode;

                if (clientErrorText == "null")
                    nodeJson["client_status"] = nullptr;
                else
                    nodeJson["client_status"] = clientErrorText;

                    nodeJson["size"] = urlCollection->getSize();

                for(auto url : *urlCollection->getUrls()){
                    nlohmann::json tmp;
                    tmp["url_text"] = url->getUrlText();
                    tmp["url"] = url->getUrl();
                    nodeJson["urls"].push_back(tmp);
                }

            }
        }

        nodes.push_back(nodeJson);
        ++nodeNum;
    }
    std::ofstream o(fullPath+folderName+".json");
    o << std::setw(4) << nodes << std::endl;
}

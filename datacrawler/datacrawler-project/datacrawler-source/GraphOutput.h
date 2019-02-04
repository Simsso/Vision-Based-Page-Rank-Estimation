//
// Created by doktorgibson on 1/26/19.
//

#ifndef DATACRAWLER_PROJECT_GRAPHOUTPUT_H
#define DATACRAWLER_PROJECT_GRAPHOUTPUT_H

#include "iostream"
#include "map"

#include "opencv2/opencv.hpp"
#include "graph/NodeElement.h"
#include "datamodules/screenshot-datamodule/Screenshot.h"
#include "datamodules/url-datamodule/UrlCollection.h"
#include "util/Logger.h"
#include "util/json.hpp"

class GraphOutput {
private:
    std::map<std::string, NodeElement *> *graph;
    std::string folderName;
    std::string outputPath = "/home/doktorgibson/Desktop/test/";
    Logger* logger;

public:
    void generateGraph();
    GraphOutput(std::map<std::string, NodeElement *>*, std::string);
    ~GraphOutput();
};


#endif //DATACRAWLER_PROJECT_GRAPHOUTPUT_H

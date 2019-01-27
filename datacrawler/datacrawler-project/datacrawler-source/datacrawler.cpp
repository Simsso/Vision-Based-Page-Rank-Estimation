#include "datacrawler.h"

/**
 * Datacrawler
 */
Datacrawler::Datacrawler() {
    logger = Logger::getInstance();
}

Datacrawler::~Datacrawler() {}

/**
 * init - Loads all user-defined DataModules and prepares Datacrawler to crawl given url
 */
void Datacrawler::init() {
    logger->info("Initialising Datacrawler !");

    logger->info("Loading general configuration!");
    numNodes = datacrawlerConfiguration.getNumNodes();

    logger->info("Number of nodes " + to_string(numNodes));

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MODULE)->createInstance());
        logger->info("Using Screenshot-DataModule ..");
    }

    if (datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(SCREENSHOT_MOBILE_MODULE)->createInstance());
        logger->info("Using ScreenshotMobile-DataModule ..");
    }

    if (datacrawlerConfiguration.getConfiguration(URL_MODULE) != nullptr) {
        dataModules.push_front(datacrawlerConfiguration.getConfiguration(URL_MODULE)->createInstance());
        logger->info("Using URL-Module ..");
    }

    logger->info("Initialising Datacrawler finished!");
}

/**
 * process - Process given url with loaded DataModules
 * @param url which should be processed
 * @return NodeElement which represents a node in the graph with all data the user defined for the graph
 */
map<string, NodeElement*>* Datacrawler::process(string url) {
    logger->info("Analysing start node!");
    logger->info("Processing <" + url + ">");
    logger->info("Running DataModules!");

    graph = new map<std::string, NodeElement*>;
    NodeElement * startNode = new NodeElement(true);
    UrlCollection * startNodeUrlCollection = nullptr;

    for (auto x: dataModules) {
        startNode->addData(x->process(url));
    }

    logger->info("<" + url + "> processed!");

    for (auto x: *startNode->getData()) {
        // get base url of starting node, which is different from domain name
        if (x->getDataModuleType() == URL_MODULE)
            startNodeUrlCollection = (UrlCollection*) x;
    }

    if (startNodeUrlCollection == nullptr) {
        // for single-node graphs, we use the passed url.
        graph->insert(make_pair(url, startNode));
        return graph;
    } else if (startNodeUrlCollection->getUrls()->empty()) {
        // for single-node graphs, we use the passed url.
        graph->insert(make_pair(url, startNode));
        return graph;
    }

    queue<pair<string, NodeElement*>> nodes;

    graph->insert(make_pair(startNodeUrlCollection->getBaseUrl(), startNode));
    nodes.push(make_pair(startNodeUrlCollection->getBaseUrl(), startNode));
    --numNodes;

    while(!nodes.empty()) {
        if(numNodes <= 0)
            break;

        vector<pair<string,NodeElement*>> newNodes = buildNodes(nodes.front().second);

        for(auto node: newNodes){
            graph->insert(node);
            nodes.push(node);
        }
        nodes.pop();
    }

    // delete arbitary edges
   for(auto node : *graph){
      auto nodeData = node.second->getData();

      for(auto entry : *nodeData){
          if(entry->getDataModuleType() != URL_MODULE)
              continue;

          vector<Url*> * urls = ((UrlCollection*)entry)->getUrls();
          urls->erase(std::remove_if(urls->begin(), urls->end(),
                                 [&](Url* url) { return (graph->find(url->getUrl()) == graph->end()); }), urls->end());
      }
   }

    for(auto node : *graph){
        auto nodeData = node.second->getData();
        logger->info("node: "+ node.first);

        for(auto entry : *nodeData){
            if(entry->getDataModuleType() != URL_MODULE)
                continue;

            vector<Url*> * urls = ((UrlCollection*)entry)->getUrls();
            for(auto url : *urls)
                logger->info("---> "+url->getUrl());
        }
    }

    return graph;
}

vector<pair<string, NodeElement*>> Datacrawler::buildNodes(NodeElement* startNode) {
    UrlCollection *urlCollection;
    NodeElement *newNode;
    vector<pair<string,NodeElement*>> newNodes;

    for (auto x: *startNode->getData()) {
        if (x->getDataModuleType() == URL_MODULE)
            urlCollection = (UrlCollection *) x;
    }

    for (auto edge : *urlCollection->getUrls()) {
        string edgeUrl = edge->getUrl();

        // check if exists in the graph
        if(graph->find(edgeUrl) != graph->end())
                continue;

        // check if we are allowed to generate more nodes
        if (numNodes <= 0)
            return newNodes;

        newNode = new NodeElement(false);

        // calculate new node
        for (auto x: dataModules) {
            newNode->addData(x->process(edgeUrl));
        }

        // insert into graph
        newNodes.push_back(make_pair(edgeUrl, newNode));
        --numNodes;
    }
    return newNodes;
}
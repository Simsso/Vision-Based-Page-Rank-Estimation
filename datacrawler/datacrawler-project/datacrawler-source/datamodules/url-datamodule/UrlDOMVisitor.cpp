//
// Created by doktorgibson on 1/13/19.
//

#include <chrono>
#include "UrlDOMVisitor.h"

UrlDOMVisitor::UrlDOMVisitor(vector<pair<string,string>>& urls, string url, int numUrls) : validUrls(urls){
    logger = Logger::getInstance();
    this->numUrls = numUrls;
    this->url = url;
}

UrlDOMVisitor::~UrlDOMVisitor() {}

void UrlDOMVisitor::Visit(CefRefPtr<CefDOMDocument> domDocument) {
    logger->info("Parsing URLs !");
    logger->info("Initial URL (passed) is " + url);
    logger->info("Crawling all URLs of the DOM !");

    calculatedUrl = domDocument.get()->GetBaseURL();

    std::smatch match;
    regex regex_baseUrl("^(https|http):\\/\\/[a-zA-Z-0-9.-]*");

    if(!regex_search(calculatedUrl, match, regex_baseUrl))
        return;

    calculatedUrl = match[0];

    logger->info("Current visited Url is "+calculatedUrl);

    queue<CefRefPtr<CefDOMNode>> aQueue = traverseDOMTree(domDocument.get()->GetBody());
    filterURL(aQueue);
    shuffleURLs();

    logger->info("Returning "+to_string(numUrls)+" URLs !");

    int numRemove = abs(numUrls - (int)validUrls.size());
    for(int i = 0; i < numRemove; i++)
        validUrls.pop_back();

}

queue<CefRefPtr<CefDOMNode>> UrlDOMVisitor::traverseDOMTree(CefRefPtr<CefDOMNode> body) {
    logger->info("Traversing DOM!");

    queue<CefRefPtr<CefDOMNode>> aQueue;
    queue<CefRefPtr<CefDOMNode>> nodeQueue;

    nodeQueue.push(body);

    // find all nodes with tag <a>
    while (!nodeQueue.empty()) {
        CefRefPtr<CefDOMNode> currentNode = nodeQueue.front();

        if (currentNode.get()->HasChildren()) {
            CefRefPtr<CefDOMNode> tmp = currentNode.get()->GetFirstChild().get();
            nodeQueue.push(tmp);

            while (tmp.get()->GetNextSibling().get() != NULL) {
                tmp = tmp.get()->GetNextSibling().get();
                nodeQueue.push(tmp);

                if (tmp.get()->IsElement())
                    if (tmp.get()->GetElementTagName() == "A")
                        aQueue.push(tmp);
            }
        }
        nodeQueue.pop();
    }
    return aQueue;
}

void UrlDOMVisitor::filterURL(queue<CefRefPtr<CefDOMNode>> &aQueue) {
    logger->info("Gathering valid Urls !");
    regex httpRegex("^http://" + url);
    regex httpsRegex("^https://" + url);
    regex wwwRegex("^http://www." + url);
    regex wwwHttpsRegex("^https://www." + url);
    regex implicitProtocolRegex("^//" + url);
    regex relativePathRegex("^/.*");
    regex anchorRegex("^#.*");
    bool calculatedUrlHasHttps = false;

    if(regex_search(calculatedUrl, httpsRegex))
       calculatedUrlHasHttps = true;

    while (!aQueue.empty()) {
        string urlText = aQueue.front().get()->GetElementInnerText();
        string url = aQueue.front().get()->GetElementAttribute("href");
        aQueue.pop();

        // TODO optimise
       if (regex_search(url, httpRegex)) {
           validUrlMap.insert(make_pair(url, urlText));

        } else if (regex_search(url, httpsRegex)) {
           validUrlMap.insert(make_pair(url, urlText));

        } else if (regex_search(url, wwwHttpsRegex)) {
           validUrlMap.insert(make_pair(url, urlText));

       } else if (regex_search(url, wwwRegex)) {
           validUrlMap.insert(make_pair(url, urlText));

       } else if (regex_search(url, implicitProtocolRegex)) {
           string newCalculatedUrl = calculatedUrlHasHttps ? "https:" + url : "http:"+ url;
           validUrlMap.insert(make_pair(newCalculatedUrl, urlText));

       } else if (regex_search(url, relativePathRegex)) {
           string newCalculatedUrl = calculatedUrl+url;
           validUrlMap.insert(make_pair(newCalculatedUrl, urlText));

       } else if (regex_search(url, anchorRegex)) {
           string newCalculatedUrl = calculatedUrl+'/'+url;
           validUrlMap.insert(make_pair(newCalculatedUrl, urlText));
       }
    }
}

void UrlDOMVisitor::shuffleURLs(){
    logger->info("Shuffling URLs!");

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    for (auto x : validUrlMap)
        validUrls.push_back(make_pair(x.first, x.second));

    shuffle(validUrls.begin(), validUrls.end(), default_random_engine(seed));
}
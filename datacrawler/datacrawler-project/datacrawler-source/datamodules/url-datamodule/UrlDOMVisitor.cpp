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
    logger->info("Passed URL is " + url);
    logger->info("Crawling all URLs of the DOM !");

    baseUrl = domDocument.get()->GetBaseURL();
    std::smatch match;
    regex regex_baseUrl("^(https|http):\\/\\/[a-zA-Z-0-9.-]*");
    regex regex_domainName("[a-zA-Z-0-9.-]*$");

    if(!regex_search(baseUrl, match, regex_baseUrl))
        return;

    baseUrl = match[0];

    if(!regex_search(baseUrl, match, regex_domainName))
        return;

    baseUrlDomainOnly = match[0];

    logger->info("Current visited base-URL is "+baseUrl);

    queue<CefRefPtr<CefDOMNode>> aQueue = traverseDOMTree(domDocument.get()->GetBody());
    filterURL(aQueue);
    shuffleURLs();

    int numRemove = 0;
    logger->info(to_string(numUrls));
    if(numUrls <= (int)validUrls.size())
        numRemove = abs(numUrls - (int)validUrls.size());

    for(int i = 0; i < numRemove; i++)
        validUrls.pop_back();

    logger->info("Returning "+to_string(validUrls.size())+" URLs !");
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
    regex httpRegex("^http://" + baseUrlDomainOnly);
    regex httpsRegex("^https://" + baseUrlDomainOnly);
    regex wwwRegex("^http://www." + baseUrlDomainOnly);
    regex wwwHttpsRegex("^https://www." + baseUrlDomainOnly);
    regex implicitProtocolRegex("^//" + baseUrlDomainOnly);
    regex relativePathRegex("^/.*");
    regex anchorRegex("^#.*");
    bool baseUrlHasHttps = false;

    if(regex_search(baseUrl, httpsRegex))
        baseUrlHasHttps = true;

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
           string newCalculatedUrl = baseUrlHasHttps ? "https:" + url : "http:"+ url;
           validUrlMap.insert(make_pair(newCalculatedUrl, urlText));

       } else if (regex_search(url, relativePathRegex)) {
           string newCalculatedUrl = baseUrl+url;
           validUrlMap.insert(make_pair(newCalculatedUrl, urlText));

       } else if (regex_search(url, anchorRegex)) {
           string newCalculatedUrl = baseUrl+'/'+url;
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
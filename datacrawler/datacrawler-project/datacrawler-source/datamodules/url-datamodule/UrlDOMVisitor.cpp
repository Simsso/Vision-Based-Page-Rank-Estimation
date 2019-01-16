//
// Created by doktorgibson on 1/13/19.
//

#include "UrlDOMVisitor.h"

UrlDOMVisitor::UrlDOMVisitor() {
    logger = Logger::getInstance();
}

UrlDOMVisitor::UrlDOMVisitor(queue<Url*>* queue) {
    logger = Logger::getInstance();
    this->validUrl = queue;
}

UrlDOMVisitor::~UrlDOMVisitor() {}

void UrlDOMVisitor::Visit(CefRefPtr<CefDOMDocument> domDocument) {
    logger->info("Parsing URLs !");
    logger->info("Base-URL is " + url);
    logger->info("Crawling all URLs of the DOM !");
    calculatedUrl = domDocument.get()->GetBaseURL();
    // TODO get only top-level domain of calculated URL
    logger->info("Calculated Url is "+calculatedUrl);

    queue<CefRefPtr<CefDOMNode>> aQueue = traverseDOMTree(domDocument.get()->GetBody());
    filterURL(aQueue);
}

queue<CefRefPtr<CefDOMNode>> UrlDOMVisitor::traverseDOMTree(CefRefPtr<CefDOMNode> body) {

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

        logger->info(url);
       if (regex_search(url, httpRegex)) {
            validUrl->push(new Url(urlText, url, false));
        } else if (regex_search(url, httpsRegex)) {
            validUrl->push(new Url(urlText, url, true));
        } else if (regex_search(url, wwwHttpsRegex)) {
            validUrl->push(new Url(urlText, url, true));
        } else if (regex_search(url, wwwRegex)) {
            validUrl->push(new Url(urlText, url, false));
        } else if (regex_search(url, implicitProtocolRegex)) {
            validUrl->push(new Url(urlText, calculatedUrlHasHttps ? "https:" + url : "http:"+ url, calculatedUrlHasHttps));
        } else if (regex_search(url, relativePathRegex)) {
           // TODO add trimmed calculated Url
            validUrl->push(new Url(urlText, calculatedUrl+url, calculatedUrlHasHttps));
        } else if (regex_search(url, anchorRegex)) {
            validUrl->push(new Url(urlText, url, calculatedUrlHasHttps));
        }
    }
}

void UrlDOMVisitor::setUrl(string url) {
    this->url = url;
}
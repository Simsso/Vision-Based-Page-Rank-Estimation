//
// Created by doktorgibson on 1/13/19.
//

#include <regex>
#include "UrlDOMVisitor.h"

UrlDOMVisitor::UrlDOMVisitor() {
    logger = Logger::getInstance();
}

UrlDOMVisitor::~UrlDOMVisitor() {}

void UrlDOMVisitor::Visit(CefRefPtr<CefDOMDocument> domDocument) {
    logger->info("Parsing URLs !");

    logger->info("Base-URL is " + url);

    logger->info("Crawling all URLs the DOM !");
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

                if(tmp.get()->IsElement())
                    if(tmp.get()->GetElementTagName() == "A")
                        aQueue.push(tmp);
            }
        }
        nodeQueue.pop();
    }

    return aQueue;
}

void UrlDOMVisitor::filterURL(queue<CefRefPtr<CefDOMNode>>& aQueue){
    regex httpRegex("(http://"+url+")");
    regex httpsRegex("(https://"+url+")");
    regex implicitProtocolRegex("(//"+url+")");
    regex relativePathRegex("(^/\\w+)");

    while(!aQueue.empty()){
        string urlText = aQueue.front().get()->GetElementInnerText();
        string url = aQueue.front().get()->GetElementAttribute("href");

        logger->info(url);

        if(regex_search(url, httpRegex)){
            logger->info("http-regex");
        } else if (regex_search(url, httpsRegex)){
            logger->info("https-regex");
        } else if (regex_search(url, implicitProtocolRegex)){
            logger->info("implicitprotocol");
        } else if (regex_search(url, relativePathRegex)){
            logger->info("relative path");
        }

        aQueue.pop();
    }
}

void UrlDOMVisitor::setUrl(string url) {
    this->url = url;
}
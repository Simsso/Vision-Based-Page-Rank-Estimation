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
    logger->info("Crawling all URLs the DOM !");
    filterURL(traverseDOMTree(domDocument.get()->GetBody()));
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
    regex httpRegex("http://" + url);
    regex httpsRegex("https://" + url);
    regex wwwRegex("http://www." + url);
    regex wwwHttpsRegex("https://www." + url);
    regex implicitProtocolRegex("//" + url);
    regex relativePathRegex("^/.*");
    regex anchorRegex("^#.*");

    while (!aQueue.empty()) {
        string urlText = aQueue.front().get()->GetElementInnerText();
        string url = aQueue.front().get()->GetElementAttribute("href");

        if (regex_search(url, httpRegex)) {
            validUrl->push(new Url(urlText, url, false));
        } else if (regex_search(url, httpsRegex)) {
            validUrl->push(new Url(urlText, url, true));
        } else if (regex_search(url, wwwHttpsRegex)) {
            validUrl->push(new Url(urlText, url, true));
        } else if (regex_search(url, wwwRegex)) {
            validUrl->push(new Url(urlText, url, false));
        } else if (regex_search(url, implicitProtocolRegex)) {
            validUrl->push(new Url(urlText, url));
        } else if (regex_search(url, relativePathRegex)) {
            validUrl->push(new Url(urlText, url));
        } else if (regex_search(url, anchorRegex)) {
            validUrl->push(new Url(urlText, url));
        }
        aQueue.pop();
    }
}

void UrlDOMVisitor::setUrl(string url) {
    this->url = url;
}
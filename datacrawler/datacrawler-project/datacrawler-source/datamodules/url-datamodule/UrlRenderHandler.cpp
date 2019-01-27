//
// Created by doktorgibson on 1/13/19.
//

#include <include/wrapper/cef_helpers.h>
#include "UrlRenderHandler.h"

UrlRenderHandler::UrlRenderHandler() {
    logger = Logger::getInstance();
}

UrlRenderHandler::~UrlRenderHandler(){}

void UrlRenderHandler::GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) {
    rect = CefRect(0, 0, 10, 10);
}

void UrlRenderHandler::OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList &dirtyRects,
                                const void *buffer, int width, int height) {
    // DO NOTHING
}
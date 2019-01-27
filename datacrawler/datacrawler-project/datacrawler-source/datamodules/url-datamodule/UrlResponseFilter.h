//
// Created by doktorgibson on 1/27/19.
//

#ifndef DATACRAWLER_PROJECT_URLRESPONSEFILTER_H
#define DATACRAWLER_PROJECT_URLRESPONSEFILTER_H

#include <include/cef_response_filter.h>
#include "../../util/Logger.h"

class UrlResponseFilter: public CefResponseFilter {
    IMPLEMENT_REFCOUNTING(UrlResponseFilter);
private:
    size_t * totalSize;
    size_t left;
    Logger* logger;

public:
    bool InitFilter() OVERRIDE;
    FilterStatus Filter(void* data_in,
                        size_t data_in_size,
                        size_t& data_in_read,
                        void* data_out,
                        size_t data_out_size,
                        size_t& data_out_written) OVERRIDE;

    UrlResponseFilter(unsigned long*);
    ~UrlResponseFilter();
};


#endif //DATACRAWLER_PROJECT_URLRESPONSEFILTER_H

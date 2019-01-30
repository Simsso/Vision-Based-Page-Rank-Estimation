//
// Created by doktorgibson on 1/27/19.
//

#include "UrlResponseFilter.h"

UrlResponseFilter::UrlResponseFilter(size_t& totalSize) : totalSize(totalSize) {
    this->logger = Logger::getInstance();
}

UrlResponseFilter::~UrlResponseFilter() {}

bool UrlResponseFilter::InitFilter() {
    return true;
}

CefResponseFilter::FilterStatus UrlResponseFilter::Filter(void* data_in,
                    size_t data_in_size,
                    size_t& data_in_read,
                    void* data_out,
                    size_t data_out_size,
                    size_t& data_out_written){

    if(data_in_size == 0) {
        data_out_written = 0;
        data_in_read = 0;
        return RESPONSE_FILTER_DONE;
    }

    data_out_written = std::min(data_in_size, data_out_size);
    data_in_read = data_out_written;
    memcpy(data_out, data_in, data_out_written);

    totalSize += data_out_written;

    if(data_in_size > data_out_size){
        return RESPONSE_FILTER_NEED_MORE_DATA;
    }

    return  RESPONSE_FILTER_DONE;
}
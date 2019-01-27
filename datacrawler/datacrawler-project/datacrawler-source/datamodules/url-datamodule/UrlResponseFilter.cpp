//
// Created by doktorgibson on 1/27/19.
//

#include "UrlResponseFilter.h"

UrlResponseFilter::UrlResponseFilter(size_t *totalSize) {
    this->totalSize = totalSize;
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

    *totalSize += data_in_size;

    if (data_in == null || dataIn.Length == 0)
    {
        dataInRead = 0;
        dataOutWritten = 0;

        return FilterStatus.Done;
    }

    if(data_out_size >= data_in_size){
        memcpy(data_out, data_in, data_in_size);
        data_in_read = data_in_size;
        data_out_written = data_in_size;
        return RESPONSE_FILTER_DONE;
    }


    memcpy(data_out, data_in, data_out_size);
    data_in_read = data_out_size;
    data_out_written = data_out_size;

    return RESPONSE_FILTER_NEED_MORE_DATA;
}
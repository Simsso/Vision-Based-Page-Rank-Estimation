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

  if (data_in == NULL) {
        data_in_read = 0;
        data_out_written = 0;

        return RESPONSE_FILTER_DONE;
    }

    //Calculate how much data we can read, in some instances dataIn.Length is
    //greater than dataOut.Length
    data_in_read = std::min(data_in_size, data_out_size);
    data_out_written = data_in_read;

    // write data out
    memcpy(data_out, data_in, data_in_read);

    //If we read less than the total amount avaliable then we need
    //return FilterStatus.NeedMoreData so we can then write the rest
    if (data_in_read < data_in_size)
    {
        return RESPONSE_FILTER_NEED_MORE_DATA;
    }

    return RESPONSE_FILTER_DONE;
}
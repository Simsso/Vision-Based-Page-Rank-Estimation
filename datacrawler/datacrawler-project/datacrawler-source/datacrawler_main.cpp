//
// Created by Samed Güner on 18.10.18.
//

#include "datacrawler.h"

Logger* Logger::instance = 0;

int main(int argc, char* argv[]){

    Datacrawler datacrawler("https://google.com/");
    datacrawler.init();
    datacrawler.process();
}

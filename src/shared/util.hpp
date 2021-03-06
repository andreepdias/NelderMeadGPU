#pragma once

#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <algorithm>

#define PI 3.1415926535897932384626433832795029

double stime(){
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
    return mlsec/1000.0;
}

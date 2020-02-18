#include <math.h>


float square1(float * p_vertex){

    float result = 0.0f;

    for(int i = 0; i < 100; i++){
        result += (p_vertex[i] * p_vertex[i]) / 100.0f;
    }

    return result;
}

float absolute_sum(float * p_vertex){

    float result = 0.0f;

    for(int i = 0; i < 100; i++){
        result += abs(p_vertex[i]) / 100.0f;
    }

    return result;
}

float square2(float * p_vertex){

    float result = 0.0f;

    for(int i = 0 ; i < 200; i++){
        result += (p_vertex[i] * p_vertex[i]) / 200;
    }

    return result;
}
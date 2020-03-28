#pragma once

#include "fast/nelderMeadFast.cuh"

#include "standard/single/nelderMeadSingle.cuh"
#include "standard/multi/nelderMeadMulti.cuh"
// #include "standard/nelderMeadShared.cuh"

#include "shared/objectiveFunctions.cuh"


NelderMeadResult nelderMead(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

    // return nelderMeadFast(parameters, h_problem_parameters);

    printf("oi oi  oi\n");
    if(parameters.optimization_type == SINGLE){
        printf("olha aquela bola\n");
        return nelderMeadSingle(parameters, h_problem_parameters, d_problem_parameters);
    }else if(parameters.optimization_type == MULTI){
        printf("BLAU\n");
        return nelderMeadMulti(parameters, h_problem_parameters, d_problem_parameters);
    }else{
        printf("tchchuca\n");
        return nelderMeadFast(parameters, h_problem_parameters);
    }
	
}
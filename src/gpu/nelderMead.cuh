#pragma once

#include "fast/nelderMeadFast.cuh"
#include "standard/single/nelderMeadSingle.cuh"
#include "standard/multi/nelderMeadMulti.cuh"

#include "shared/objectiveFunctions.cuh"


NelderMeadResult nelderMead(NelderMead &parameters, std::ofstream &output, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

    // return nelderMeadFast(parameters, output, h_problem_parameters);

    if(parameters.optimization_type == SINGLE){
        return nelderMeadSingle(parameters, h_problem_parameters, d_problem_parameters);
    }else if(parameters.optimization_type == MULTI){
        return nelderMeadMulti(parameters, h_problem_parameters, d_problem_parameters);
    }else{
        return nelderMeadFast(parameters, output, h_problem_parameters);
    }
}
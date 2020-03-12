#ifndef NELMIN_H
#define NELMIN_H

#include "util.cuh"
#include "nelderMeadSingle.cuh"
#include "nelderMeadMulti.cuh"


NelderMeadResult nelderMead(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

    if(parameters.multi_vertexes){
        return nelderMeadMulti(parameters, h_problem_parameters, d_problem_parameters);
    }else{
        return nelderMeadSingle(parameters, h_problem_parameters, d_problem_parameters);
    }
	
}

#endif
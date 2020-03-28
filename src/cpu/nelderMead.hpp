#pragma once

#include "standard/nelderMeadStandard.hpp"
#include "fast/nelderMeadFast.hpp"

#include "shared/objectiveFunctions.hpp"

NelderMeadResult nelderMead(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

    if(parameters.optimization_type == FAST){
        return nelderMeadFast(parameters, h_problem_parameters);
    }else{
        return nelderMeadSingle(parameters, h_problem_parameters);
    }
	
}
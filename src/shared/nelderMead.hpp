#pragma once

#include "util.hpp"

enum OptimizationTypeEnum {
    SINGLE,
    MULTI,
    FAST
};


struct NelderMead{

    OptimizationTypeEnum optimization_type;
    int p;

    int executions_number;
    int iterations_number;
    int evaluations_number;

    int dimension;

	float step;
	float reflection_coef;
	float expansion_coef;
	float contraction_coef;
	float shrink_coef;

	int evaluations_used;
	
	float * p_start;
    std::vector<float> start;

	float * p_simplex;	

	float * p_centroid;
	float * p_reflection;
	float * p_expansion;
	float * p_contraction;

	std::pair<float, int> * p_objective_function;
	std::pair<float, int> * p_obj_reflection;
	std::pair<float, int> * p_obj_expansion;
	std::pair<float, int> * p_obj_contraction;

    std::vector< std::vector<float> > starting_points;

    bool show_best_vertex;

};

struct NelderMeadResult{

    float best;    
    float elapsed_time;
    
    std::vector<float> best_vertex;

    int evaluations_used;
    int latest_improvement;
};


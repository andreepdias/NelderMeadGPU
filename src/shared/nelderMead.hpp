#pragma once

#include "util.hpp"

enum ProblemEnum {
    NO_PROBLEM,
    BENCHMARK,
    AB_OFF_LATTICE
};

enum BenchmarkProblemEnum {
    NONE,
    SQUARE,
    SUM
};

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

	ProblemEnum problem_type;
	BenchmarkProblemEnum benchmark_problem;
    
    std::vector< std::vector<float> > starting_points;

    bool show_best_vertex;

};

struct NelderMeadResult{

    float best;    
    std::vector<float> best_vertex;
    int evaluations_used;
};


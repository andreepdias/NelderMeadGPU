#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <math.h>

#define PI 3.1415926535897932384626433832795029

double stime(){
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
    return mlsec/1000.0;
}


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

struct ABOffLattice{

	int protein_length;
	const char * aminoacid_sequence;

};

struct NelderMead{

	int iterations_number;

	int dimension;

	float step;
	float reflection_coef;
	float expansion_coef;
	float contraction_coef;
	float shrink_coef;
	
	float * p_start;

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

};

#endif
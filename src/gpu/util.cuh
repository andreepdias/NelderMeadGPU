#ifndef __UTIL_H
#define __UTIL_H

/* Thrust imports */
//#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/fill.h>


#include <cooperative_groups.h>


/* C++ imports */
#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <vector>

__constant__ char aminoacid_sequence[150];

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

    int evaluations_used;

	float step;
	float reflection_coef;
	float expansion_coef;
	float contraction_coef;
    float shrink_coef;
    
	float * p_start;

	ProblemEnum problem_type;
	BenchmarkProblemEnum benchmark_problem;

};

struct NelderMeadResult{

    float best;    
    std::vector<float> best_vertex;
    int evaluations_used;
};

#endif
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

    bool multi_vertexes;
    int p;

    int executions_number;
    int iterations_number;
	int dimension;

    int evaluations_used;

	float step;
	float reflection_coef;
	float expansion_coef;
	float contraction_coef;
    float shrink_coef;
    
    float * p_start;
    std::vector<float> start;

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

/* ---------------------------------------- READING INPUT FILES ---------------------------------------- */

void readInputData(NelderMead &parameters, std::ifstream &input_data){
    std::vector< std::vector<float> > starting_points(parameters.executions_number, std::vector<float> (parameters.dimension));

    for(int i = 0; i < parameters.executions_number; i++){
        for(int j = 0; j < parameters.dimension; j++){
            input_data >> starting_points[i][j];
        }
    }

    parameters.starting_points = starting_points;
}

void readInputBenchmark(NelderMead &parameters, std::ifstream &input_file){
  
    switch(parameters.benchmark_problem){
        case SQUARE:
            parameters.dimension = std::min(parameters.dimension, 200);
            break;
        case SUM:
            parameters.dimension = std::min(parameters.dimension, 100);
            break;
    }

    std::string dimension_file = (parameters.dimension < 100) ? std::to_string(100) : std::to_string(parameters.dimension);
    std::string path = "resources/data_inputs/data" + dimension_file + "dimension.txt";
    std::ifstream input_data(path.c_str());

    readInputData(parameters, input_data);
}

void readInputABOffLatttice(NelderMead &parameters, ABOffLattice * &parametersAB, std::ifstream &input_file){
    
    parametersAB = new ABOffLattice();

    std::string protein_name, protein_chain;
    std::vector<float> angles;

    input_file >> protein_name;
    input_file >> protein_chain;

    float angle;
    while(input_file >> angle){
        angles.push_back(angle * PI / 180.0f);
    }

    parameters.start = angles;
    parameters.p_start = &parameters.start[0];

    parameters.dimension = angles.size();

    (*parametersAB).aminoacid_sequence = protein_chain.c_str();
    (*parametersAB).protein_length = protein_chain.size();
}

bool readInput(NelderMead &parameters, std::ifstream &input_file, ABOffLattice * &parametersAB){

    int executions_number, iterations_number, dimension;

    std::string s;
    input_file >> s;

    if(s == "BENCHMARK"){
        parameters.problem_type = BENCHMARK;
        
        input_file >> s;

        if(s == "SQUARE"){
            parameters.benchmark_problem = SQUARE;
        }else if(s == "SUM"){
            parameters.benchmark_problem = SUM;
        }else{
            printf("A funcao objetivo do Benhmark nao foi especificada corretamente. Tente:\nSQUARE ou SUM.\n");
            return false;
        }
    }else if(s == "ABOFFLATTICE"){
        parameters.problem_type = AB_OFF_LATTICE;
    }else{
        printf("O tipo do problema nao foi especifiado. Tente:\nBENCHMARK ou ABOFFLATTICE.\n");
        return false;
    }

    input_file >> s;

    if(s == "SINGLE"){
        parameters.multi_vertexes = false;
    }else if(s == "MULTI"){
        parameters.multi_vertexes = true;

        int p;
        input_file >> p;
        parameters.p = p;
    }else{
        printf("E necessario especificar a variacao do Nelder Mead. Tente:\nSINGLE ou MULTI.\n");
        return false;
    }
    
    input_file >> executions_number;
    input_file >> iterations_number;
    input_file >> dimension;

    parameters.executions_number = executions_number;
    parameters.iterations_number = iterations_number;
    parameters.dimension = dimension;

    if(parameters.problem_type == BENCHMARK){
       readInputBenchmark(parameters, input_file);
    }else if(parameters.problem_type == AB_OFF_LATTICE){
        readInputABOffLatttice(parameters, parametersAB, input_file);
    }

    input_file >> s;

    if(s == "SHOW_BEST_VERTEX"){
        parameters.show_best_vertex = true;
    }else{
        parameters.show_best_vertex = false;
    }
    
    return true;
}

#endif
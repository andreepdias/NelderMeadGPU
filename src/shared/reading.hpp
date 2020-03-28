#pragma once

#include "util.hpp"
#include "abOffLattice.hpp"

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

    (*parametersAB).protein_name = protein_name;
    (*parametersAB).aa_sequence = protein_chain;
    (*parametersAB).aminoacid_sequence = (*parametersAB).aa_sequence.c_str();
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
        parameters.optimization_type = SINGLE;
    }else if(s == "MULTI"){
        parameters.optimization_type = MULTI;

        int p;
        input_file >> p;
        parameters.p = p;
    }else if(s == "FAST"){
        parameters.optimization_type = FAST;
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
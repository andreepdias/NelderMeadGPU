#pragma once

#include "util.hpp"
#include "abOffLattice.hpp"

void readInput(std::ifstream &input_file, OptimizationTypeEnum &optimization_type, int &executions, int &evaluations, int &proteins_evalued, int &p){

    std::string s;

    input_file >> s;

    if(s == "SINGLE"){
        optimization_type = SINGLE;
    }else if(s == "MULTI"){
        optimization_type = MULTI;
        input_file >> p;
    }else{
        optimization_type = FAST;
    }

    input_file >> executions;
    input_file >> evaluations;
    input_file >> proteins_evalued;
}

void readInputProteins(std::ifstream &input_file, int &evaluations, int &p, OptimizationTypeEnum &optimization_type, std::vector<NelderMead> &parameters, std::vector<ABOffLattice*> &parametersAB){

    int n = parameters.size();

    std::string name, chain;
    std::vector<float> angles;
    int psl, dim;
    float x;

    for(int i = 0; i < n; i++){
        input_file >> name;
        input_file >> chain;
        psl = chain.size();
        dim = psl * 2 - 5;

        angles.resize(dim);
        for(int j = 0; j < dim; j++){
            input_file >> x;
            angles[j] = (x * PI / 180.0f);
            // angles[j] = x;
        }

        parameters[i].optimization_type = optimization_type;
        parameters[i].dimension = dim;
        parameters[i].evaluations_number = evaluations;
        parameters[i].start = angles;
        parameters[i].p_start = &parameters[i].start[0];
        parameters[i].show_best_vertex = false;
        parameters[i].p = p;

        parametersAB[i] = new ABOffLattice();
        (*parametersAB[i]).protein_length = psl;
        (*parametersAB[i]).protein_name = name;
        (*parametersAB[i]).aa_sequence = chain;
        (*parametersAB[i]).aminoacid_sequence = (*parametersAB[i]).aa_sequence.c_str();
    }
}

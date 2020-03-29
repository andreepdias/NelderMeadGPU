#pragma once

#include "util.hpp"
#include "abOffLattice.hpp"

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

    int executions_number, evaluations_number, iterations_number, dimension;

    std::string s;
    input_file >> s;

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
    input_file >> evaluations_number;
    input_file >> dimension;

    parameters.executions_number = executions_number;
    parameters.evaluations_number = evaluations_number;
    parameters.dimension = dimension;
    
    readInputABOffLatttice(parameters, parametersAB, input_file);

    input_file >> s;

    if(s == "SHOW_BEST_VERTEX"){
        parameters.show_best_vertex = true;
    }else{
        parameters.show_best_vertex = false;
    }
    
    return true;
}
#pragma once

#include "util.hpp"

void printParameters(NelderMead &parameters, ABOffLattice * & parametersAB){
    
    printf("\n-------------------- PARAMETERS --------------------\n");
    printf("Executions: %d\n", parameters.executions_number);
    printf("Iterations: %d\n", parameters.iterations_number);
    printf("Dimension:  %d\n", parameters.dimension);
    printf("----------------------------------------------------\n");
    
    if(parameters.problem_type == AB_OFF_LATTICE){
        
        printf("Protein Name:   "); std::cout << parametersAB->protein_name << std::endl;
        printf("Protein Length: %d\n", parametersAB->protein_length);
        printf("Protein Chain:  %s\n", parametersAB->aminoacid_sequence);
        printf("----------------------------------------------------\n");
    }
}

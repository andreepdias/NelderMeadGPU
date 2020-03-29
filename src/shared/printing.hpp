#pragma once

#include "util.hpp"

void printParameters(NelderMead &parameters, ABOffLattice * & parametersAB){
    
    printf("     *General Parameters:*\n");
    // printf("Executions: %d\n", parameters.executions_number);
    printf("Evaluations: %d\n", parameters.evaluations_number);
    // printf("Dimension:  %d\n", parameters.dimension);
    printf("\n");
      
}

void printProtein(ABOffLattice * & parametersAB) {
    printf("     *Protein Parameters:*\n");
    printf("Protein Name:   "); std::cout << parametersAB->protein_name << std::endl;
    printf("Protein Length: %d\n", parametersAB->protein_length);
    printf("Protein Chain:  %s\n\n", parametersAB->aminoacid_sequence);
}

void printResult(NelderMeadResult &result, float elapsed_time){
        printf("Best: %.7f\n", result.best);
        printf("Evaluations: %d\n", result.evaluations_used);
        printf("Elapsed Time: %.7f\n", elapsed_time);
        printf("Latest Improvement: %d\n\n", result.latest_improvement);

        printf("---------------------------------------------------------\n\n");
}
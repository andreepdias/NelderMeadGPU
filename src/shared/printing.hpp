#pragma once

#include "util.hpp"


void printResults(std::vector<NelderMeadResult> &results){

    int n = results.size();

    float mean_time = 0.0f;
    float mean_obj = 0.0f;

    float best, worst;
    best = worst = results[0].best;

    printf(" \n    -> Results:\n\n");
    for(int i = 0; i < n; i++){
        printf("   Run #%d\n", i + 1);
        printf("Elapsed Time: %.4f\n", results[i].elapsed_time);
        printf("Obj Function: %.7f\n", results[i].best);
        printf("\n");
        mean_time += results[i].elapsed_time;
        mean_obj += results[i].best;
        best = std::min(best, results[i].best);
        worst = std::max(best, results[i].best);
    }

    mean_time /= n;
    mean_obj /= n;

    float standard_devation_time = 0.0f;
    float standard_devation_obj = 0.0f;

    for(int i = 0; i < n; i++){
        standard_devation_time += pow(results[i].elapsed_time - mean_time, 2);
        standard_devation_obj += pow(results[i].best - mean_obj, 2);
    }

    standard_devation_time = sqrt(standard_devation_time / n);
    standard_devation_obj = sqrt(standard_devation_obj / n);

    printf("   -All Runs:-\n");
    printf("Mean Elapsed Time: %.4f\n", mean_time);
    printf("Standard Deviation Elapsed Time: %.4f\n\n", standard_devation_time);

    printf("Mean Objective Function: %.4f\n", mean_obj);
    printf("Standard Deviation Objective Function: %.4f\n\n", standard_devation_obj);

    printf("Best of all runs: %.7f\n", best);
    printf("Worst of all runs: %.7f\n", worst);

    printf("-----------------------------------------------\n");


}

void printParameters(int &executions, int &proteins_evalued, int &evaluations){

    printf("Running NelderMead algorithm...\n\n");
    printf("Executions:  %d\n", executions);
    printf("Evaluations:  %d\n", evaluations);
    printf("Proteins Evalued: %d\n\n", proteins_evalued);
    printf("-----------------------------------------------\n");

}

void printProteinParameters(NelderMead &parameters, ABOffLattice * parametersAB){

    printf("Protein Name: "); std::cout << (*parametersAB).protein_name << std::endl;
    printf("Protein Chain: "); std::cout << (*parametersAB).aa_sequence << std::endl;
    printf("Protein Length: "); std::cout << (*parametersAB).protein_length << std::endl;
    printf("Dimension: "); std::cout << parameters.dimension << std::endl << std::endl;

}

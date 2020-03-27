#include "util.cuh"
#include "nelmin.cuh"

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

int main() {

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;
    
    ABOffLattice * parametersAB;

    std::ifstream input_file("resources/inputs/input.txt");

    if(!readInput(parameters, input_file, parametersAB)){
        return 1;
    }

    printParameters(parameters, parametersAB);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    if(parameters.problem_type == BENCHMARK){

        std::vector<NelderMeadResult> results(parameters.executions_number);
        
        for(int i = 0; i < parameters.executions_number; i++){
            parameters.p_start = &parameters.starting_points[i][0];

            cudaEventRecord(start);

            results[i] = nelderMead(parameters);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);

            printf(" - Execution %d:\n", i + 1);

            printf("Best: %.7f\n", results[i].best);
            
            if(parameters.show_best_vertex){
                printf("Best Vertex:\n");

                for(int j = 0; j < parameters.dimension; j++){
                    printf("%.7f ", results[i].best_vertex[j]);
                }
                printf("\n");
            }

            printf("Evaluations: %d\n", results[i].evaluations_used);
            printf("Elapsed Time: %.7f\n", elapsed_time / 1000.0);
        }

        float mean = 0.0f;
        for(int i = 0; i < parameters.executions_number; i++){
            mean += results[i].best;
        }
        mean /= parameters.executions_number;

        printf("\nMean of Best vertexes: %.7f\n", mean);

    }else if(parameters.problem_type == AB_OFF_LATTICE){

        char aa_sequence[150];
        memset(aa_sequence, 0, sizeof(char) * 150);
        strcpy(aa_sequence, (*parametersAB).aminoacid_sequence);
        cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

        (*parametersAB).d_aminoacid_position.resize((*parametersAB).protein_length * 3);
        (*parametersAB).p_aminoacid_position = thrust::raw_pointer_cast(&(*parametersAB).d_aminoacid_position[0]);
        
        (*parametersAB).h_aminoacid_position.resize((*parametersAB).protein_length * 3);

        cudaEventRecord(start);

        NelderMeadResult result = nelderMead(parameters, (void*) parametersAB);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        printf("Best: %.7f\n", result.best);

        if(parameters.show_best_vertex){
            printf("Best Vertex:\n");
            
            for(int i = 0; i < parameters.dimension; i++){
                printf("%.7f ", result.best_vertex[i]);
            }
        }
        printf("\nEvaluations: %d\n", result.evaluations_used);
        printf("Elapsed Time: %.7f\n", elapsed_time / 1000.0);

    }

}


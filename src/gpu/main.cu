
#include "util.cuh"
#include "nelderMead.cuh"


int main() {

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;
    
    ABOffLattice * parametersAB;

    std::ifstream input_file("resources/inputs/input.txt");

    if(!readInput(parameters, input_file, parametersAB)){
        return 1;
    }

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

            printf("Execution %d: Best %.7f - Elapsed Time: %.7f\n", i + 1, results[i].best, elapsed_time / 1000.0);
        }

        float mean = 0.0f;
        for(int i = 0; i < parameters.executions_number; i++){
            mean += results[i].best;
        }
        mean /= parameters.executions_number;

        printf("\nMean: %.7f\n", mean);

    }else if(parameters.problem_type == AB_OFF_LATTICE){

        ABOffLattice * d_parametersAB;

        cudaMalloc(&d_parametersAB, sizeof(ABOffLattice));
        cudaMemcpy(d_parametersAB, parametersAB, sizeof(ABOffLattice), cudaMemcpyHostToDevice);

        char aa_sequence[150];
        memset(aa_sequence, 0, sizeof(char) * 150);
        strcpy(aa_sequence, aminoacid_sequence);
        cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

        cudaEventRecord(start);

        NelderMeadResult result = nelderMead(parameters, (void*) parametersAB, (void*) d_parametersAB );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        printf("Best: %.7f\nVertex: ", result.best);
        
        for(int i = 0; i < parameters.dimension; i++){
            printf("%.7f ", result.best_vertex[i]);
        }
        printf("\nEvaluations: %d\n", result.evaluations_used);
        printf("Elapsed Time: %.7f\n", elapsed_time / 1000.0);

    }

}


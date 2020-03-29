#include "../shared/nelderMead.hpp"
#include "../shared/util.hpp"
#include "../shared/reading.hpp"
#include "../shared/printing.hpp"

#include "../shared/abOffLattice.hpp"
#include "nelderMead.cuh"

int main() {

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;
    
    ABOffLattice * parametersAB;

    std::ifstream input_file("resources/inputs/input.txt");
    std::ofstream output_file("resources/outputs/output.txt");

    if(!readInput(parameters, input_file, parametersAB)){
        return 1;
    }

    printParameters(parameters, parametersAB);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;


    ABOffLattice * d_parametersAB;

    cudaMalloc(&d_parametersAB, sizeof(ABOffLattice));
    cudaMemcpy(d_parametersAB, parametersAB, sizeof(ABOffLattice), cudaMemcpyHostToDevice);

    char aa_sequence[150];
    memset(aa_sequence, 0, sizeof(char) * 150);
    strcpy(aa_sequence, (*parametersAB).aminoacid_sequence);
    cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

    cudaEventRecord(start);

    NelderMeadResult result = nelderMead(parameters, output_file, (void*) parametersAB, (void*) d_parametersAB);

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

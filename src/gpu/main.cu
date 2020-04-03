#include "../shared/nelderMead.hpp"
#include "../shared/util.hpp"
#include "../shared/reading.hpp"
#include "../shared/printing.hpp"

#include "../shared/abOffLattice.hpp"
#include "nelderMead.cuh"

void run(int &executions, int &proteins_evalued, std::vector<NelderMead> &parameters, std::vector<ABOffLattice*> &parametersAB, int d = 0){

    std::ofstream output_plot_file;
    std::string path;
    
    float elapsed_time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printParameters(executions, proteins_evalued, parameters[0].evaluations_number);

    for(int k = 0; k < proteins_evalued; k++){

        // if(d >= parameters[k].dimension){
            // continue;
        // }

        ABOffLattice * d_parametersAB;
        cudaMalloc(&d_parametersAB, sizeof(ABOffLattice));
        cudaMemcpy(d_parametersAB, parametersAB[k], sizeof(ABOffLattice), cudaMemcpyHostToDevice);

        char aa_sequence[150];
        memset(aa_sequence, 0, sizeof(char) * 150);
        strcpy(aa_sequence, (*parametersAB[k]).aminoacid_sequence);
        cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));


        printProteinParameters(parameters[k], parametersAB[k]);

        std::vector<NelderMeadResult> results(executions);

        for(int i = 0; i < executions; i++){

            printf("Running execution %d...\n", i + 1);

            path = "resources/outputs/plot_" + std::to_string(k) + "_" + (*parametersAB[k]).protein_name + "_" + std::to_string(i) + ".txt";
            // output_plot_file.open(path.c_str(), std::ofstream::out);

            cudaEventRecord(start);
            results[i] = nelderMead(parameters[k], output_plot_file, (void*) parametersAB[k], (void*) d_parametersAB );
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time, start, stop);
            results[i].elapsed_time = elapsed_time / 1000.0f;

            // output_plot_file.close();
        }

        printResults(results);
    }
}

int main() {

    OptimizationTypeEnum optimization_type;
    int executions, evaluations, proteins_evalued, p;

    std::ifstream input_file("resources/inputs/input.txt");
    readInput(input_file, optimization_type, executions, evaluations, proteins_evalued, p);

    std::vector<NelderMead> parameters(proteins_evalued);
    std::vector<ABOffLattice*> parametersAB(proteins_evalued);

    readInputProteins(input_file, evaluations, p, optimization_type, parameters, parametersAB);
    
    // for(int i = 1; i <= 64; i *= 2){

        // printf("-*-*-*-*-*-*-*-*-*-*-*-*-*- P == %d --*-*-*-*-*-*-*-*-*-*-*-*-*\n", i);

        // for(int  j = 0; j < proteins_evalued; j++){
            // parameters[j].p = i;
        // }

    run(executions, proteins_evalued, parameters, parametersAB);
    // }


}

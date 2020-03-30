#include "../shared/nelderMead.hpp"
#include "../shared/abOffLattice.hpp"
#include "../shared/util.hpp"
#include "../shared/reading.hpp"
#include "../shared/printing.hpp"

#include "nelderMead.hpp"

void run(int &executions, int &proteins_evalued, std::vector<NelderMead> &parameters, std::vector<ABOffLattice*> &parametersAB){

    std::ofstream output_plot_file;
    std::string path;
    double start, stop, elapsed_time;

    printParameters(executions, proteins_evalued, parameters[0].evaluations_number);

    for(int k = 0; k < proteins_evalued; k++){

        printProteinParameters(parameters[k], parametersAB[k]);

        std::vector<NelderMeadResult> results(executions);

        for(int i = 0; i < executions; i++){

            printf("Running execution %d...\n", i + 1);

            path = "resources/outputs/plot_" + std::to_string(k) + "_" + (*parametersAB[k]).protein_name + "_" + std::to_string(i) + ".txt";
            // output_plot_file.open(path.c_str(), std::ofstream::out);

            start = stime();
            results[i] = nelderMead(parameters[k], output_plot_file, (void*) parametersAB[k] );
            stop = stime();
            results[i].elapsed_time = stop - start;

            // output_plot_file.close();
        }

        printResults(results);
    }
}

int main(int argc, char * argv[]){

    OptimizationTypeEnum optimization_type;
    int executions, evaluations, proteins_evalued, p;

    std::ifstream input_file("resources/inputs/proteins.txt");
    readInput(input_file, optimization_type, executions, evaluations, proteins_evalued, p);

    std::vector<NelderMead> parameters(proteins_evalued);
    std::vector<ABOffLattice*> parametersAB(proteins_evalued);

    readInputProteins(input_file, evaluations, p, optimization_type, parameters, parametersAB);
    
    run(executions, proteins_evalued, parameters, parametersAB);

    return 0;
}

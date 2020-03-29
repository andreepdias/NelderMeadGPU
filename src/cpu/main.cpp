#include "../shared/nelderMead.hpp"
#include "../shared/abOffLattice.hpp"
#include "../shared/util.hpp"
#include "../shared/reading.hpp"
#include "../shared/printing.hpp"

#include "nelderMead.hpp"

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

void readInputProteins(std::ifstream &input_file, int &evaluations, OptimizationTypeEnum &optimization_type, std::vector<NelderMead> &parameters, std::vector<ABOffLattice*> &parametersAB){

    int n = parameters.size();

    std::string name, chain;
    std::vector<float> angles;
    int psl, dim, p;
    float x;

    for(int i = 0; i < n; i++){
        input_file >> name;
        input_file >> psl >> dim;
        input_file >> chain;

        angles.resize(dim);
        for(int j = 0; j < dim; j++){
            input_file >> x;
            angles[j] = (x * PI / 180.0f);
        }

        parameters[i].optimization_type = optimization_type;
        parameters[i].dimension = dim;
        parameters[i].evaluations_number = evaluations;
        parameters[i].start = angles;
        parameters[i].p_start = &parameters[i].start[0];
        parameters[i].show_best_vertex = false;

        parametersAB[i] = new ABOffLattice();
        (*parametersAB[i]).protein_length = psl;
        (*parametersAB[i]).protein_name = name;
        (*parametersAB[i]).aa_sequence = chain;
        (*parametersAB[i]).aminoacid_sequence = (*parametersAB[i]).aa_sequence.c_str();
    }
}

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

    printf("Running NelderMead algorithm on CPU...\n\n");
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

            path = "resources/outputs/plot_" + std::to_string(k) + "_" + (*parametersAB[i]).protein_name + "_" + std::to_string(i) + ".txt";
            output_plot_file.open(path.c_str(), std::ofstream::out);

            start = stime();
            results[i] = nelderMead(parameters[k], output_plot_file, (void*) parametersAB[k] );
            stop = stime();
            results[i].elapsed_time = stop - start;

            output_plot_file.close();
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

    readInputProteins(input_file, evaluations, optimization_type, parameters, parametersAB);
    
    run(executions, proteins_evalued, parameters, parametersAB);

    return 0;
}

#include "../shared/nelderMead.hpp"
#include "../shared/abOffLattice.hpp"
#include "../shared/util.hpp"
#include "../shared/reading.hpp"
#include "../shared/printing.hpp"

#include "nelderMead.hpp"

void readInput(std::ifstream &input_file, OptimizationTypeEnum &optimization_type, int &executions, int &evaluations, int &proteins_evalued){

    std::string s;

    input_file >> s;

    if(s == "SINGLE"){
        optimization_type = SINGLE;
    }else if(s == "MULTI"){
        optimization_type = MULTI;
    }else{
        optimization_type = FAST;
    }

    input_file >> executions;
    input_file >> evaluations;
    input_file >> proteins_evalued;
}


int main(int argc, char * argv[]){

    OptimizationTypeEnum optimization_type;
    int executions, evaluations, proteins_evalued;

    std::ifstream input_file("resources/inputs/input.txt");
    readInput(input_file, optimization_type, executions, evaluations, proteins_evalued);

    std::vector<NelderMead> parameters(proteins_evalued);
    std::vector<ABOffLattice*> parameters(proteins_evalued);

    std::ofstream output_file;

    // std::ofstream output_file("resources/outputs/output.txt");
    // std::ifstream input_file("resources/inputs/input.txt");

    // if(!readInput(parameters, input_file, parametersAB)){
        // return 1;
    // }
    // printParameters(parameters, parametersAB);

    double start, stop, elapsed_time;


    float psl, n, x;
    std::string name, chain;
    

    std::ofstream output_file;
    std::ifstream protein_file("resources/inputs/proteins.txt");
    if(!protein_file.is_open()){
        printf("File does not exist\n");
    }

    protein_file >> parameters.evaluations_number;
    parameters.optimization_type = FAST;
    parameters.show_best_vertex = false;

    printParameters(parameters, parametersAB);

    int k = 0;
    while(protein_file >> name){
        k++;
        parametersAB = new ABOffLattice();

        (*parametersAB).protein_name = name;

        protein_file >> (*parametersAB).protein_length >> parameters.dimension;
        protein_file >> (*parametersAB).aa_sequence;

        (*parametersAB).aminoacid_sequence = (*parametersAB).aa_sequence.c_str();

        std::vector<float> angles;

        for(int i  = 0; i < parameters.dimension; i++){
            protein_file >> x;
            angles.push_back(x * PI / 180.0f);
        }
        parameters.start = angles;
        parameters.p_start = &parameters.start[0];

        std::string path = "resources/outputs/output_" + std::to_string(k)  + "_" + (*parametersAB).protein_name + ".txt";
        output_file.open (path.c_str(), std::ofstream::out);

        printProtein(parametersAB);

        start = stime();

        NelderMeadResult result = nelderMead(parameters, output_file, (void*) parametersAB );

        stop = stime();
        elapsed_time = stop - start;

        output_file.close();

        printResult(result, elapsed_time) ;
    }

       
    // start = stime();

    // NelderMeadResult result = nelderMead(parameters, output_file, (void*) parametersAB );

    // stop = stime();
    // elapsed_time = stop - start;

    // printf("Best: %.7f\n", result.best);
    // printf("Evaluations: %d\n", result.evaluations_used);
    // printf("Elapsed Time: %.7f\n", elapsed_time);

    // if(parameters.show_best_vertex){
    //     printf("Best Vertex:\n");

    //     for(int i = 0; i < parameters.dimension; i++){
    //         printf("%.7f ", result.best_vertex[i]);
    //     }
    // }

    return 0;
}

#include "util.hpp"
#include "nelmin.hpp"

int main(){

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;

    std::ifstream input_file("input.txt");

    std::string s;
    input_file >> s;

    if(s == "BENCHMARK"){
        parameters.problem_type = BENCHMARK;

        input_file >> s;

        if(s == "SQUARE1"){
            parameters.benchmark_problem = SQUARE1;
        }else if(s == "SQUARE2"){
            parameters.benchmark_problem = SQUARE2;
        }else if(s == "SUM"){
            parameters.benchmark_problem = SUM;
        }else{
            printf("A funcao objetivo do Benhmark nao foi especificada corretamente. Tente:\nSQUARE1, SQUARE2 ou SUM.\n");
            return 1;
        }

    }else if(s == "ABOFFLATTICE"){
        parameters.problem_type = AB_OFF_LATTICE;
    }else{
        printf("O tipo do problema nao foi especifiado. Tente:\nBENCHMARK ou ABOFFLATTICE.\n");
        return 1;
    }


    if(parameters.problem_type == BENCHMARK){

    }else{
        ABOffLattice * parametersAB;

        std::string protein_name, protein_chain;
        std::vector<float> angles;

        int iterations_number;

        input_file >> iterations_number;
        input_file >> protein_name;
        input_file >> protein_chain;

        float angle;
        while(input_file >> angle){
            angles.push_back(angle * PI / 180.0f);
        }

        (*parametersAB).aminoacid_sequence = protein_chain.c_str();
        (*parametersAB).protein_length = protein_chain.size();

        parameters.iterations_number = iterations_number;
        parameters.dimension = angles.size();
        parameters.p_start = &angles[0];

        nelderMead(parameters, (void*) parametersAB );
    }
    

    return 0;
        

}
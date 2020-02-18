#include "util.hpp"
#include "nelmin.hpp"

enum ProblemEnum {
    NO_PROBLEM,
    BENCHMARK,
    AB_OFF_LATTICE
};

enum BenchmarkProblemEnum {
    NONE,
    SQUARE1,
    SQUARE2,
    SUM
};

int main(){

    ProblemEnum problem_type = NO_PROBLEM;
    BenchmarkProblemEnum benchmark_problem = NONE;

    std::ifstream input_file("input.txt");

    std::string s;
    input_file >> s;

    if(s == "BENCHMARK"){
        problem_type = BENCHMARK;

        input_file >> s;

        if(s == "SQUARE1"){
            benchmark_problem = SQUARE1;
        }else if(s == "SQUARE2"){
            benchmark_problem = SQUARE2;
        }else if(s == "SUM"){
            benchmark_problem = SUM;
        }else{
            printf("A funcao objetivo do Benhmark nao foi especificada corretamente. Tente:\nSQUARE1, SQUARE2 ou SUM.\n");
            return 1;
        }

    }else if(s == "ABOFFLATTICE"){
        problem_type = AB_OFF_LATTICE;
    }else{
        printf("O tipo do problema nao foi especifiado. Tente:\nBENCHMARK ou ABOFFLATTICE.\n");
        return 1;
    }

    NelderMead parameters;

    if(problem_type == BENCHMARK){

    }else{
        ABOffLattice parametersAB;

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

        int protein_length = protein_chain.size();
        int dimension = angles.size();

    }
    
    nelderMeaaaad(1, 1, NULL, "", 0);

    return 0;
        

}
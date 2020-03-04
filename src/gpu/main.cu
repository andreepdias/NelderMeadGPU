
#include "util.cuh"
#include "nelderMeadMulti.cuh"

int main() {

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;

    std::ifstream input_file("resources/inputs/input.txt");

    std::string s;
    input_file >> s;

    if(s == "BENCHMARK"){
        parameters.problem_type = BENCHMARK;
        
        input_file >> s;

        if(s == "SQUARE"){
            parameters.benchmark_problem = SQUARE;
        }else if(s == "SUM"){
            parameters.benchmark_problem = SUM;
        }else{
            printf("A funcao objetivo do Benhmark nao foi especificada corretamente. Tente:\nSQUARE ou SUM.\n");
            return 1;
        }
    }else if(s == "ABOFFLATTICE"){
        parameters.problem_type = AB_OFF_LATTICE;
    }else{
        printf("O tipo do problema nao foi especifiado. Tente:\nBENCHMARK ou ABOFFLATTICE.\n");
        return 1;
    }
    
    int dimension, iterations_number, executions_number;

    if(parameters.problem_type == BENCHMARK){

        input_file >> dimension;
        input_file >> iterations_number;
        input_file >> executions_number;

        parameters.iterations_number = iterations_number;

        switch(parameters.benchmark_problem){
            case SQUARE:
                parameters.dimension = std::min(dimension, 200);
                break;
            case SUM:
                parameters.dimension = std::min(dimension, 100);
                break;
        }

        std::string dimension_file = (parameters.dimension < 100) ? std::to_string(100) : std::to_string(parameters.dimension);

        std::string path = "resources/data_inputs/data" + dimension_file + "dimension.txt";
        std::ifstream input_data(path.c_str())        ;

        std::vector< std::vector<float> > start_point(executions_number, std::vector<float> (parameters.dimension));

        for(int i = 0; i < executions_number; i++){
            for(int j = 0; j < parameters.dimension; j++){
                input_data >> start_point[i][j];
            }
        }

        std::vector<NelderMeadResult> results(executions_number);

        for(int i = 0; i < executions_number; i++){
            parameters.p_start = &start_point[i][0];

            results[i] = nelderMead(parameters);

            printf("Execucao %d: %.7f\n", i + 1, results[i].best);
        }

        float mean = 0.0f;
        for(int i = 0; i < executions_number; i++){
            mean += results[i].best;
        }
        mean /= executions_number;

        printf("\nMedia: %.7f\n", mean);


    }else if(parameters.problem_type == AB_OFF_LATTICE){

        ABOffLattice * parametersAB = new ABOffLattice();

        std::string protein_name, protein_chain;
        std::vector<float> angles;

        input_file >> iterations_number;
        input_file >> executions_number;
        input_file >> protein_name;
        input_file >> protein_chain;

        float angle;
        while(input_file >> angle){
            angles.push_back(angle * PI / 180.0f);
        }

        parameters.iterations_number = iterations_number;
        parameters.dimension = angles.size();
        parameters.p_start = &angles[0];

        (*parametersAB).aminoacid_sequence = protein_chain.c_str();
        (*parametersAB).protein_length = protein_chain.size();

        ABOffLattice * d_parametersAB;
        cudaMalloc(&d_parametersAB, sizeof(ABOffLattice));
        cudaMemcpy(d_parametersAB, parametersAB, sizeof(ABOffLattice), cudaMemcpyHostToDevice);

        char aa_sequence[150];
        memset(aa_sequence, 0, sizeof(char) * 150);
        strcpy(aa_sequence, protein_chain.c_str());
        cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

        NelderMeadResult result = nelderMead(parameters, (void*) parametersAB, (void*) d_parametersAB );

        printf("Best: %.7f\nVertex: ", result.best);

        for(int i = 0; i < parameters.dimension; i++){
            printf("%.7f ", result.best_vertex[i]);
        }
        printf("\nEvaluations: %d\n", result.evaluations_used);

    }

}
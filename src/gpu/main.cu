
#include "util.cuh"
#include "nelderMead.cuh"

void readInputData(NelderMead &parameters, std::ifstream &input_data){
    std::vector< std::vector<float> > starting_points(parameters.executions_number, std::vector<float> (parameters.dimension));

    for(int i = 0; i < parameters.executions_number; i++){
        for(int j = 0; j < parameters.dimension; j++){
            input_data >> starting_points[i][j];
        }
    }

    parameters.starting_points = starting_points;
}

void readInputBenchmark(NelderMead &parameters, std::ifstream &input_file){
  
    switch(parameters.benchmark_problem){
        case SQUARE:
            parameters.dimension = std::min(parameters.dimension, 200);
            break;
        case SUM:
            parameters.dimension = std::min(parameters.dimension, 100);
            break;
    }

    std::string dimension_file = (parameters.dimension < 100) ? std::to_string(100) : std::to_string(parameters.dimension);
    std::string path = "resources/data_inputs/data" + dimension_file + "dimension.txt";
    std::ifstream input_data(path.c_str());

    readInputData(parameters, input_data);
}

void readInputABOffLatttice(NelderMead &parameters, ABOffLattice * &parametersAB, std::ifstream &input_file){
    
    parametersAB = new ABOffLattice();

    std::string protein_name, protein_chain;
    std::vector<float> angles;

    input_file >> protein_name;
    input_file >> protein_chain;

    float angle;
    while(input_file >> angle){
        angles.push_back(angle * PI / 180.0f);
    }

    parameters.p_start = &angles[0];

    (*parametersAB).aminoacid_sequence = protein_chain.c_str();
    (*parametersAB).protein_length = protein_chain.size();
}

bool readInput(NelderMead &parameters, std::ifstream &input_file, ABOffLattice * &parametersAB){

    int executions_number, iterations_number, dimension;

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
            return false;
        }
    }else if(s == "ABOFFLATTICE"){
        parameters.problem_type = AB_OFF_LATTICE;
    }else{
        printf("O tipo do problema nao foi especifiado. Tente:\nBENCHMARK ou ABOFFLATTICE.\n");
        return false;
    }

    input_file >> s;

    if(s == "SINGLE"){
        parameters.multi_vertexes = false;
    }else if(s == "MULTI"){
        parameters.multi_vertexes = true;

        int p;
        input_file >> p;
        parameters.p = p;
    }else{
        printf("E necessario especificar a variacao do Nelder Mead. Tente:\nSINGLE ou MULTI.\n");
        return false;
    }
    
    input_file >> executions_number;
    input_file >> iterations_number;
    input_file >> dimension;

    parameters.executions_number = executions_number;
    parameters.iterations_number = iterations_number;
    parameters.dimension = dimension;

    if(parameters.problem_type == BENCHMARK){
       readInputBenchmark(parameters, input_file);
    }else if(parameters.problem_type == AB_OFF_LATTICE){
        readInputABOffLatttice(parameters, parametersAB, input_file);
    }
    
    return true;
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

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time  = 0.0f;

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
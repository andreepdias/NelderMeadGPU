#include "util.hpp"
#include "nelmin.hpp"

int main(){

    NelderMead parameters;
    parameters.benchmark_problem = NONE;
    parameters.problem_type = NO_PROBLEM;

    ABOffLattice * parametersAB;

    std::ifstream input_file("resources/inputs/input.txt");

    if(!readInput(parameters, input_file, parametersAB)){
        return 1;
    }

    double start, stop, elapsed_time;

    if(parameters.problem_type == BENCHMARK){

        std::vector<NelderMeadResult> results(parameters.executions_number);

        for(int i = 0; i < parameters.executions_number; i++){
            parameters.p_start = &parameters.starting_points[i][0];

            start = stime();

            results[i] = nelderMead(parameters);

            stop = stime();
            elapsed_time = stop - start;

            printf(" - Execution %d:\n", i + 1);

            printf("Best: %.7f\n", results[i].best);

            if(parameters.show_best_vertex){
                printf("Best Vertex:\n");

                for(int j = 0; j < parameters.dimension; j++){
                    printf("%.7f ", results[i].best_vertex[j]);
                }
                printf("\n");
            }

            printf("Evaluations: %d\n", results[i].evaluations_used);
            printf("Elapsed Time: %.7f\n", elapsed_time);
        }

        float mean = 0.0f;
        for(int i = 0; i < parameters.executions_number; i++){
            mean += results[i].best;
        }
        mean /= parameters.executions_number;

        printf("\nMean of Best vertexes: %.7f\n", mean);

    }else if(parameters.problem_type == AB_OFF_LATTICE){
        
        start = stime();

        NelderMeadResult result = nelderMead(parameters, (void*) parametersAB );

        stop = stime();
        elapsed_time = stop - start;

        printf("Best: %.7f\n", result.best);

        if(parameters.show_best_vertex){
            printf("Best Vertex:\n");

            for(int i = 0; i < parameters.dimension; i++){
                printf("%.7f ", result.best_vertex[i]);
            }
        }
        printf("\nEvaluations: %d\n", result.evaluations_used);
        printf("Elapsed Time: %.7f\n", elapsed_time);
    }

    return 0;
}
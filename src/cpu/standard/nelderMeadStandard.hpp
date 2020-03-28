#pragma once

#include "../shared/objectiveFunctions.hpp"

NelderMeadResult nelderMeadSingle(NelderMead &parameters, void * problem_parameters);

void printVertex(int dimension, float * p_vertex, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", p_vertex[i]);
	}
	printf("\n\n");
}

void printSimplex(int dimension, float * p_simplex, std::pair<float, int> * p_obj, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		int stride = p_obj[i].second * dimension;
		printf("%2d. ", p_obj[i].second + 1);
		for(int j = 0; j < dimension; j++){
			printf("%.5f ", p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjFunction(int dimension, std::pair<float, int> * p_obj, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. %.10f\n", p_obj[i].second + 1, p_obj[i].first);
	}
	printf("\n");
}

void printSingleObjFunction(std::pair<float, int> * p_obj, const char * msg){
	printf("%s:\n", msg);
	printf("%2d. %.10f\n\n", p_obj[0].second + 1, p_obj[0].first);
}

void nelderMead_initialize(NelderMead &p){

	for(int i = 0; i < p.dimension + 1; i++){
		for(int j = 0; j < p.dimension; j++){

			if(i != j){
				p.p_simplex[i * p.dimension + j] = p.p_start[j];
			}else{
				p.p_simplex[i * p.dimension + j] = p.p_start[j] + p.step;
			}
		}
	}
}

void nelderMead_calculate(NelderMead & p, void * problem_p, int number_evalueted_vertexes, float * p_simplex, std::pair<float, int> * p_objective_function){

	if(p.problem_type == AB_OFF_LATTICE){
		calculateABOffLattice(p, problem_p, number_evalueted_vertexes,p_simplex, p_objective_function);
	}else if(p.problem_type == BENCHMARK){

		switch(p.benchmark_problem){
			case SQUARE:
				calculateSquare(p, number_evalueted_vertexes, p_simplex, p_objective_function);
				break;
			case SUM:
				calculateAbsoluteSum(p, number_evalueted_vertexes, p_simplex, p_objective_function);
				break;
		}
	}
}

void nelderMead_centroid(NelderMead &p){

	/* Constrói todos os elementos do centróide */
	for(int i = 0; i < p.dimension; i++){
 
		float sum = 0.0f;

		/* Para cada elemento, percorrer todos os vértices com exceção do pior */
		for(int j = 0; j < p.dimension; j++){
			sum += p.p_simplex[p.p_objective_function[j].second * p.dimension + i];
		}

		p.p_centroid[i] = sum / p.dimension;
	}
}

void nelderMead_reflection(NelderMead &p){

	for(int i  = 0; i < p.dimension; i++){
		p.p_reflection[i] = p.p_centroid[i] + p.reflection_coef * (p.p_centroid[i] - p.p_simplex[p.p_objective_function[p.dimension].second * p.dimension + i]);
	}

}

void nelderMead_expansion(NelderMead &p){

	for(int i = 0; i < p.dimension; i++){

		p.p_expansion[i] = p.p_reflection[i] + p.expansion_coef * (p.p_reflection[i] - p.p_centroid[i]);
	}
}

void nelderMead_contraction_firstCase(NelderMead &p){
	
	for(int i = 0; i < p.dimension; i++){
		p.p_contraction[i] = p.p_centroid[i] + p.contraction_coef * (p.p_reflection[i] - p.p_centroid[i]);
	}
}

void nelderMead_contraction_secondCase(NelderMead &p){

	for(int i = 0; i < p.dimension; i++){
		p.p_contraction[i] = p.p_centroid[i] + p.contraction_coef * (p.p_simplex[ p.p_objective_function[p.dimension].second * p.dimension + i ] - p.p_centroid[i]);
	}

}

void nelderMead_contraction(NelderMead &p){

	if(p.p_obj_reflection[0].first < p.p_objective_function[p.dimension].first){
		nelderMead_contraction_firstCase(p);
	}else{
		nelderMead_contraction_secondCase(p);
	}
}

void nelderMead_shrink(NelderMead &p){

	int stride_a0 = p.p_objective_function[0].second * p.dimension;

	for(int i = 1; i < p.dimension + 1; i++){

		int stride = p.p_objective_function[i].second * p.dimension;

		for(int j = 0; j < p.dimension; j++){
			p.p_simplex[stride + j] = p.shrink_coef * p.p_simplex[stride_a0 + j] + (1.0f - p.shrink_coef) * p.p_simplex[stride + j];
		}
	}
}

void nelderMead_replacement(NelderMead &p, float * p_new_vertex, std::pair<float, int> * p_obj){

	int stride = p.p_objective_function[p.dimension].second * p.dimension;

	for(int i = 0; i < p.dimension; i++){
		p.p_simplex[stride + i] = p_new_vertex[i];
	}

	p.p_objective_function[p.dimension].first = p_obj[0].first;

}

void printEvaluationsUpdate(NelderMead &p, int add){

	printf("*-*Total evaluations during update: %d. Right now adding: %d\n", p.evaluations_used, add);

}

void nelderMead_update(int k, NelderMead &p, void * problem_parameters, int * p_count){

	if(p.p_obj_reflection[0].first < p.p_objective_function[0].first){

		// /*c*/ p_count[0] += 1;
		
		nelderMead_expansion(p);		
		nelderMead_calculate(p, problem_parameters, 1, p.p_expansion, p.p_obj_expansion);
		// /*e*/ p.evaluations_used += 1;


		if(p.p_obj_expansion[0].first < p.p_objective_function[0].first){
			nelderMead_replacement(p, p.p_expansion, p.p_obj_expansion);
		}else{
			nelderMead_replacement(p, p.p_reflection, p.p_obj_reflection);
		}


	}else if(p.p_obj_reflection[0].first < p.p_objective_function[p.dimension - 1].first){

		// /*c*/ p_count[1] += 1;

		nelderMead_replacement(p, p.p_reflection, p.p_obj_reflection);
	}else{
		nelderMead_contraction(p);
		nelderMead_calculate(p, problem_parameters, 1, p.p_contraction, p.p_obj_contraction);
		// /*e*/ p.evaluations_used += 1;

		if(p.p_obj_contraction[0].first < p.p_objective_function[p.dimension].first){

			// /*c*/ p_count[2] += 1;

			nelderMead_replacement(p, p.p_contraction, p.p_obj_contraction);
		}else{

			// /*c*/ p_count[3] += 1;

			nelderMead_shrink(p);
			nelderMead_calculate(p, problem_parameters, p.dimension + 1, p.p_simplex, p.p_objective_function);
			// /*e*/ p.evaluations_used += p.dimension + 1;

			
		}
	}
}

void printEvaluationsHost(int total, int add){

	printf("***Total evaluations: %d. Now adding: %d\n", total, add);

}

NelderMeadResult nelderMeadSingle(NelderMead &parameters, void * problem_parameters = NULL){

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 1.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;

	std::vector<float> simplex(dimension * (dimension + 1));

	std::vector<float> centroid(dimension);
	std::vector<float> reflection(dimension);
	std::vector<float> expansion(dimension);
	std::vector<float> contraction(dimension);

	std::vector<int> count(4);

	std::vector<std::pair<float, int> > objective_function(dimension + 1);
	std::vector<std::pair<float, int> > obj_reflection(1);
	std::vector<std::pair<float, int> > obj_expansion(1);
	std::vector<std::pair<float, int> > obj_contraction(1);

	parameters.p_simplex = &simplex[0];
	
	parameters.p_centroid = &centroid[0];
	parameters.p_reflection = &reflection[0];
	parameters.p_expansion = &expansion[0];
	parameters.p_contraction = &contraction[0];

	parameters.p_objective_function = &objective_function[0];
	parameters.p_obj_reflection = &obj_reflection[0];
	parameters.p_obj_expansion = &obj_expansion[0];
	parameters.p_obj_contraction = &obj_contraction[0];

	int * p_count = &count[0];

	
	nelderMead_initialize(parameters);

	nelderMead_calculate(parameters, problem_parameters, dimension + 1, parameters.p_simplex, parameters.p_objective_function);
	// /*e*/ parameters.evaluations_used += dimension + 1;
	std::sort(objective_function.begin(), objective_function.end());

	for(int i = 0; i < parameters.iterations_number; i++){

		nelderMead_centroid(parameters);

		nelderMead_reflection(parameters);
		nelderMead_calculate(parameters, problem_parameters, 1, parameters.p_reflection, parameters.p_obj_reflection);
		// /*e*/ parameters.evaluations_used += 1;

		nelderMead_update(i, parameters, problem_parameters, p_count);

		std::sort(objective_function.begin(), objective_function.end());
	}

	NelderMeadResult result;

	result.best = objective_function[0].first;
	std::vector<float> best_vertex(simplex.begin() + objective_function[0].second * dimension, simplex.begin() + objective_function[0].second * dimension + dimension);
	result.best_vertex = best_vertex;
	result.evaluations_used = parameters.evaluations_used;

	// /*c*/ printf("case 1: %d, case 2: %d, case 3: %d, case 4: %d\n", p_count[0], p_count[1], p_count[2], p_count[3]);

	return result;

}
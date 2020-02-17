#ifndef NELMIN_H
#define NELMIN_H

#include "util.hpp"


struct NelderMead{

	int iterations_number;

	int dimension;
	int protein_length;

	const char * aminoacid_sequence;

	float step;
	float reflection_coef;
	float expansion_coef;
	float contraction_coef;
	float shrink_coef;
	
	float * start_point;

	float * p_simplex;	

	float * p_centroid;
	float * p_reflection;
	float * p_expansion;
	float * p_contraction;

	std::pair<float, int> * p_objective_function;
	std::pair<float, int> * p_obj_reflection;
	std::pair<float, int> * p_obj_expansion;
	std::pair<float, int> * p_obj_contraction;

};


void printStart(NelderMead &p){
	printf("Start:\n");
	for(int i = 0; i < p.dimension; i++){
		printf("%.5f ", p.start_point[i]);
	}
	printf("\n\n");
}

void printInitialize(NelderMead &p){
	printf("Initialize:\n");
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. ", i + 1);
		for(int j = 0; j < p.dimension; j++){
			int stride = i * p.dimension;
			printf("%.5f ", p.p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printShrink(NelderMead &p){
	printf("Shrink Case 3b (contraction worst than worst vertex):\n");
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. ", i + 1);
		for(int j = 0; j < p.dimension; j++){
			int stride = i * p.dimension;
			printf("%.5f ", p.p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printPreShrink(NelderMead &p){
	printf("Pre Shrink:\n");
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. ", i + 1);
		for(int j = 0; j < p.dimension; j++){
			int stride = i * p.dimension;
			printf("%.5f ", p.p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printReplacement(NelderMead &p, std::string msg){
	std::cout << "Replacement" << " " << msg << ":\n";
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. ", i + 1);
		for(int j = 0; j < p.dimension; j++){
			int stride = i * p.dimension;
			printf("%.5f ", p.p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjectiveFunction(NelderMead &p){
	printf("Objective Function:\n");
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. %.10f\n", p.p_objective_function[i].second + 1, p.p_objective_function[i].first);
	}
	printf("\n");
}

void printObjectiveFunctionSorted(NelderMead &p){
	printf("Objective Function Sorted:\n");
	for(int i = 0; i < p.dimension + 1; i++){
		printf("%2d. %.10f\n", p.p_objective_function[i].second + 1, p.p_objective_function[i].first);
	}
	printf("\n");
}

void printCentroid(NelderMead &p){

	printf("Centroid:\n");
	for(int i = 0; i < p.dimension; i++){
		printf("%5f ", p.p_centroid[i]);
	}
	printf("\n\n");
}

void printReflection(NelderMead &p){

	printf("Reflection:\n");
	for(int i = 0; i < p.dimension; i++){
		printf("%5f ", p.p_reflection[i]);
	}
	printf("\n\n");
}

void printObjectiveFunctionReflection(NelderMead &p){
	printf("Objective Function Relection:\n");
	printf("%2d. %.10f\n\n", p.p_obj_reflection[0].second + 1, p.p_obj_reflection[0].first);
}

void printExpansion(NelderMead &p){

	printf("Expansion:\n");
	for(int i = 0; i < p.dimension; i++){
		printf("%5f ", p.p_expansion[i]);
	}
	printf("\n\n");
}

void printObjectiveFunctionExpansion(NelderMead &p){
	printf("Objective Function Expansion:\n");
	printf("%2d. %.10f\n\n", p.p_obj_expansion[0].second + 1, p.p_obj_expansion[0].first);
}

void printContraction(NelderMead &p){

	printf("Contraction:\n");
	for(int i = 0; i < p.dimension; i++){
		printf("%5f ", p.p_contraction[i]);
	}
	printf("\n\n");
}

void printObjectiveFunctionContraction(NelderMead &p){
	printf("Objective Function Contraction:\n");
	printf("%2d. %.10f\n\n", p.p_obj_contraction[0].second + 1, p.p_obj_contraction[0].first);
}

void nelderMead_initialize(NelderMead &p){

	for(int i = 0; i < p.dimension + 1; i++){
		for(int j = 0; j < p.dimension; j++){

			if(i != j){
				p.p_simplex[i * p.dimension + j] = p.start_point[j];
			}else{
				p.p_simplex[i * p.dimension + j] = p.start_point[j] + p.step;
			}
		}
	}
}

void nelderMead_calculate(NelderMead &p, int number_evalueted_vertexes, float * p_simplex, std::pair<float, int> * p_objective_function){

	std::vector<float> aminoacid_position(p.protein_length * 3);


	for(int k = 0; k < number_evalueted_vertexes; k++){
		/* Existem dimension + 1 vértices, com dimension elementos cada, stride acessa o vértice da iteração k */
		int stride = k * p.dimension;

		for(int i = 0; i < p.protein_length - 2; i++){

			aminoacid_position[0] = 0.0f;
			aminoacid_position[0 + p.protein_length] = 0.0f;
			aminoacid_position[0 + p.protein_length * 2] = 0.0f;

			aminoacid_position[1] = 0.0f;
			aminoacid_position[1 + p.protein_length] = 1.0f;
			aminoacid_position[1 + p.protein_length * 2] = 0.0f;

			aminoacid_position[2] = cosf(p_simplex[stride + 0]);
			aminoacid_position[2 + p.protein_length] = sinf(p_simplex[stride + 0]) + 1.0f;
			aminoacid_position[2 + p.protein_length * 2] = 0.0f;

			for(int j = 3; j < p.protein_length; j++){
				aminoacid_position[j] = aminoacid_position[j - 1] + cosf(p_simplex[stride + j - 2]) * cosf(p_simplex[stride + j + p.protein_length - 5]); // j - 3 + p.protein_length - 2
				aminoacid_position[j + p.protein_length] = aminoacid_position[j - 1 + p.protein_length] + sinf(p_simplex[stride + j - 2]) * cosf(p_simplex[stride + j + p.protein_length - 5]);
				aminoacid_position[j + p.protein_length * 2] = aminoacid_position[j - 1 + p.protein_length * 2] + sinf(p_simplex[stride + j + p.protein_length - 5]);
			}
		}

		float sum = 0.0f;

		for(int i = 0; i < p.protein_length - 2; i++){
			sum += (1.0f - cosf(p_simplex[stride + i])) / 4.0f;
		}

		float c, d, dx, dy, dz;

		for(int i = 0; i < p.protein_length - 2; i++){
			for(int j = i + 2; j < p.protein_length; j++){
				if(p.aminoacid_sequence[i] == 'A' && p.aminoacid_sequence[j] == 'A')
					c = 1.0;
				else if(p.aminoacid_sequence[i] == 'B' && p.aminoacid_sequence[j] == 'B')
					c = 0.5;
				else
					c = -0.5;

				dx = aminoacid_position[i] - aminoacid_position[j];
				dy = aminoacid_position[i + p.protein_length] - aminoacid_position[j + p.protein_length];
				dz = aminoacid_position[i + p.protein_length * 2] - aminoacid_position[j + p.protein_length * 2];
				d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );

				sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
			}
		}

		p_objective_function[k].first = sum;
		p_objective_function[k].second = k;
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

	printf("Reflection: %.5f, Worst: %.5f\n", p.p_obj_reflection[0].first, p.p_objective_function[p.dimension].first);
	if(p.p_obj_reflection[0].first < p.p_objective_function[p.dimension].first){
		printf("First case contraction\n");
		nelderMead_contraction_firstCase(p);
	}else{
		printf("Second case contraction\n");
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

void nelderMead_update(NelderMead &p){

	if(p.p_obj_reflection[0].first < p.p_objective_function[0].first){
		
		nelderMead_expansion(p);
		printExpansion(p);
		
		nelderMead_calculate(p, 1, p.p_expansion, p.p_obj_expansion);
		printObjectiveFunctionExpansion(p);

		if(p.p_obj_expansion[0].first < p.p_objective_function[0].first){
			nelderMead_replacement(p, p.p_expansion, p.p_obj_expansion);
			printReplacement(p, "Case 1a (expansion better than best vertex)");
		}else{
			nelderMead_replacement(p, p.p_reflection, p.p_obj_reflection);
			printReplacement(p, "Case 1b (reflection better than best vertex)");
		}

	}else if(p.p_obj_reflection[0].first < p.p_objective_function[p.dimension - 1].first){
		nelderMead_replacement(p, p.p_reflection, p.p_obj_reflection);
		printReplacement(p, "Case 2 (reflection better than second worst vertex)");
	}else{
		nelderMead_contraction(p);
		printContraction(p);
		nelderMead_calculate(p, 1, p.p_contraction, p.p_obj_contraction);
		printObjectiveFunctionContraction(p);

		if(p.p_obj_contraction[0].first < p.p_objective_function[p.dimension].first){
			nelderMead_replacement(p, p.p_contraction, p.p_obj_contraction);
			printReplacement(p, "Case 3a (contraction better than worst vertex)");
		}else{
			printPreShrink(p);
			nelderMead_shrink(p);
			printShrink(p);
			nelderMead_calculate(p, p.dimension + 1, p.p_simplex, p.p_objective_function);
		}
	}
}

void nelderMead(int dimension, int protein_length, float start[], const char aa_sequence[], int iterations_number){

	NelderMead parameters;

	parameters.iterations_number = iterations_number;

	parameters.dimension = dimension;
	parameters.protein_length = protein_length;

	parameters.aminoacid_sequence = aa_sequence;
	
	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 1.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	std::vector<float> simplex(dimension * (dimension + 1));

	std::vector<float> centroid(dimension);
	std::vector<float> reflection(dimension);
	std::vector<float> expansion(dimension);
	std::vector<float> contraction(dimension);

	std::vector<std::pair<float, int> > objective_function(dimension + 1);
	std::vector<std::pair<float, int> > obj_reflection(1);
	std::vector<std::pair<float, int> > obj_expansion(1);
	std::vector<std::pair<float, int> > obj_contraction(1);

	parameters.start_point = start;
	
	parameters.p_simplex = &simplex[0];
	
	parameters.p_centroid = &centroid[0];
	parameters.p_reflection = &reflection[0];
	parameters.p_expansion = &expansion[0];
	parameters.p_contraction = &contraction[0];

	parameters.p_objective_function = &objective_function[0];
	parameters.p_obj_reflection = &obj_reflection[0];
	parameters.p_obj_expansion = &obj_expansion[0];
	parameters.p_obj_contraction = &obj_contraction[0];

	printStart(parameters);

	nelderMead_initialize(parameters);
	printInitialize(parameters);

	nelderMead_calculate(parameters, dimension + 1, parameters.p_simplex, parameters.p_objective_function);
	printObjectiveFunction(parameters);
	std::sort(objective_function.begin(), objective_function.end());
	printObjectiveFunctionSorted(parameters);

	for(int i = 0; i < parameters.iterations_number; i++){

		nelderMead_centroid(parameters);
		printCentroid(parameters);

		nelderMead_reflection(parameters);
		printReflection(parameters);
		nelderMead_calculate(parameters, 1, parameters.p_reflection, parameters.p_obj_reflection);
		printObjectiveFunctionReflection(parameters);

		nelderMead_update(parameters);

		printObjectiveFunction(parameters);
		std::sort(objective_function.begin(), objective_function.end());
		printObjectiveFunctionSorted(parameters);

		printf("------------------ END ITERATION %d ------------------\n\n", i + 1);
	}

}
 #endif

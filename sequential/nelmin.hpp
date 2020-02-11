#ifndef NELMIN_H
#define NELMIN_H

#include "util.hpp"


void nelderMead_initialize(int dimension, float * p_simplex, float step, float * start){

	for(int i = 0; i < dimension + 1; i++){
		for(int j = 0; j < dimension; j++){

			if(i != j){
				p_simplex[i * dimension + j] = start[j];
			}else{
				p_simplex[i * dimension + j] = start[j] + step;
			}
		}
	}
}

void nelderMead_calculate(int protein_length, int dimension, float * p_simplex, float * p_objective_function, const char aminoacid_sequence[]){

	std::vector<float> aminoacid_position(protein_length * 3);

	for(int k = 0; k < dimension + 1; k++){
		/* Existem dimension + 1 vértices, com dimension elementos cada, stride acessa o vértice da iteração k */
		int stride = k * dimension;

		for(int i = 0; i < protein_length - 2; i++){

			aminoacid_position[0] = 0.0f;
			aminoacid_position[0 + protein_length] = 0.0f;
			aminoacid_position[0 + protein_length * 2] = 0.0f;

			aminoacid_position[1] = 0.0f;
			aminoacid_position[1 + protein_length] = 1.0f;
			aminoacid_position[1 + protein_length * 2] = 0.0f;

			aminoacid_position[2] = cosf(p_simplex[stride + 0]);
			aminoacid_position[2 + protein_length] = sinf(p_simplex[stride + 0]) + 1.0f;
			aminoacid_position[2 + protein_length * 2] = 0.0f;

			for(int j = 3; j < protein_length; j++){
				aminoacid_position[j] = aminoacid_position[j - 1] + cosf(p_simplex[stride + j - 2]) * cosf(p_simplex[stride + j + protein_length - 5]); // j - 3 + protein_length - 2
				aminoacid_position[j + protein_length] = aminoacid_position[j - 1 + protein_length] + sinf(p_simplex[stride + j - 2]) * cosf(p_simplex[stride + j + protein_length - 5]);
				aminoacid_position[j + protein_length * 2] = aminoacid_position[j - 1 + protein_length * 2] + sinf(p_simplex[stride + j + protein_length - 5]);
			}
		}

		float sum = 0.0f;

		for(int i = 0; i < protein_length - 2; i++){
			sum += (1.0f - cosf(p_simplex[stride + i])) / 4.0f;
		}

		float c, d, dx, dy, dz;

		for(int i = 0; i < protein_length - 2; i++){
			for(int j = i + 2; j < protein_length; j++){
				if(aminoacid_sequence[i] == 'A' && aminoacid_sequence[j] == 'A')
					c = 1.0;
				else if(aminoacid_sequence[i] == 'B' && aminoacid_sequence[j] == 'B')
					c = 0.5;
				else
					c = -0.5;

				dx = aminoacid_position[i] - aminoacid_position[j];
				dy = aminoacid_position[i + protein_length] - aminoacid_position[j + protein_length];
				dz = aminoacid_position[i + protein_length * 2] - aminoacid_position[j + protein_length * 2];
				d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );

				sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
			}
		}

		p_objective_function[k] = sum;
	}

}

void nelderMead_centroid(float * p_centroid, float * p_simplex, uint * p_indexes, const int dimension, const int p){

}

void nelderMead_reflection(float * p_simplex_reflected, float * p_centroid, float * p_simplex, uint * p_indexes, int dimension, int p, float reflection_coef){

}

void nelderMead_update(float * p_simplex_reflected, float * p_centroid, float * p_simplex, uint * p_indexes, float * p_objective_function, float * p_objective_function_reflected, int dimension, int p, float reflection_coef){

}

void printStart(int dimension, float start[]){
	printf("Start:\n");
	for(int i = 0; i < dimension; i++){
		printf("%.3f ", start[i]);
	}
	printf("\n\n");
}

void printInitialize(float dimension, float * p_simplex){
	printf("Initialize:\n");
	for(int i = 0; i < dimension + 1; i++){
		for(int j = 0; j < dimension; j++){
			int stride = i * dimension;
			printf("%.3f ", p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjectiveFunction(float dimension, float *  p_objective_function){
	printf("Objective Function:\n");
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. %.10f\n", i + 1, p_objective_function[i]);
	}
	printf("\n");
}

void nelderMead(int dimension, int protein_length, float start[], const char aa_sequence[]){

	const float step = 1.0f;
	const int n = dimension;

	std::vector<float> simplex(n * (n + 1));
	std::vector<float> objective_function(n + 1);
	std::vector<float> simplex_reflected(n);


	float * p_objective_function = &objective_function[0];
	float * p_simplex = &simplex[0];
	float * p_simplex_reflected = &simplex_reflected[0];


	nelderMead_initialize(dimension, p_simplex, step, start);

	printStart(dimension, start);

	printInitialize(dimension, p_simplex);

	nelderMead_calculate(protein_length, dimension, p_simplex, p_objective_function, aa_sequence);

	printObjectiveFunction(dimension, p_objective_function);



}
 #endif

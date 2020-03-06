#ifndef OBJ_FUNCTION_H
#define OBJ_FUNCTION_H


#include "util.hpp"

float calculateSquare(NelderMead &p, int number_evalueted_vertexes, float * p_simplex, std::pair<float, int> * p_objective_function){

    float result;

	for(int i = 0; i < number_evalueted_vertexes; i++){
		
		result = 0.0f;
		int stride = i * p.dimension;

		for(int j = 0; j < p.dimension; j++){
			result += (p_simplex[stride + j] * p_simplex[stride +j]) / 100.0f;
		}

		p_objective_function[i].first = result;
		p_objective_function[i].second = i;
	}

    return result;
}

float calculateAbsoluteSum(NelderMead &p, int number_evalueted_vertexes, float * p_simplex, std::pair<float, int> * p_objective_function){

    float result;

	for(int i = 0; i < number_evalueted_vertexes; i++){
		
		result = 0.0f;
		int stride = i * p.dimension;

		for(int j = 0; j < p.dimension; j++){
			result += abs(p_simplex[stride + j]) / 100.0f;
		}

		p_objective_function[i].first = result;
		p_objective_function[i].second = i;
	}

    return result;
}

void calculateABOffLattice(NelderMead &p, void * problem_p, int number_evalueted_vertexes, float * p_simplex, std::pair<float, int> * p_objective_function){

    ABOffLattice * parametersAB = (ABOffLattice*)problem_p;
    int protein_length = (*parametersAB).protein_length;

	std::vector<float> aminoacid_position(protein_length * 3);

	for(int k = 0; k < number_evalueted_vertexes; k++){
		/* Existem dimension + 1 vértices, com dimension elementos cada, stride acessa o vértice da iteração k */
		int stride = k * p.dimension;

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
				if((*parametersAB).aminoacid_sequence[i] == 'A' && (*parametersAB).aminoacid_sequence[j] == 'A')
					c = 1.0;
				else if((*parametersAB).aminoacid_sequence[i] == 'B' && (*parametersAB).aminoacid_sequence[j] == 'B')
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

		p_objective_function[k].first = sum;
		p_objective_function[k].second = k;
    }
}

#endif
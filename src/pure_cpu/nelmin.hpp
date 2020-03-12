#ifndef NELMIN_H
#define NELMIN_H

#include "util.h"
#include "util.hpp"


// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html

void nelderMead_initialize(int dimension, float * p_simplex, float * p_start, float step){

	for(int i = 0; i < dimension + 1; i++){

		int stride = i * dimension;

		for(int j = 0; j < dimension; j++){

			if(i != j){
				p_simplex[stride + j] = p_start[j];
			}else{
				p_simplex[stride + j] = p_start[j] + step;
			}
		}
	}
}

void nelderMead_calculateSimplex(float (*fn)(float*), int dimension, int &evaluations_used, float * p_simplex, float * p_obj_function){

for(int i =  0; i < dimension + 1; i++){
		p_obj_function[i] = (*fn)(p_simplex + (i * dimension));
		evaluations_used += 1;
	}
}


void nelderMead_findBest(int dimension, float &best, int &index_best, float * p_obj_function){

	best = p_obj_function[0];
	index_best = 0;

	for(int i =  1; i < dimension + 1; i++){
		if(p_obj_function[i] < best){
			best = p_obj_function[i];
			index_best = i;
		}
	}
}

void nelderMead_findWorst(int dimension, float &worst, int &index_worst, float * p_obj_function){

	worst = p_obj_function[0];
	index_worst = 0;
	
	for (int i = 1; i < dimension + 1; i++ ) {
		if ( worst < p_obj_function[i] ) { 
			worst = p_obj_function[i]; 
			index_worst = i; 
		}
	}
}

void nelderMead_centroid(int dimension, int index_worst, float * p_simplex, float * p_centroid){

	float sum;
	for (int i = 0; i < dimension; i++ ) {
		sum = 0.0;
		
		for (int j = 0; j < dimension + 1; j++ ) { 
			sum += p_simplex[j *  dimension + i];
		}
		sum -= p_simplex[index_worst * dimension + i];
		p_centroid[i] = sum / dimension;
	}
}

void nelderMead_reflection(int dimension, int index_worst, float reflection_coef, float * p_simplex, float * p_centroid, float * p_reflection){

	for (int i = 0; i < dimension; i++ ) {
		p_reflection[i] = p_centroid[i] + reflection_coef * ( p_centroid[i] - p_simplex[i + index_worst * dimension] );
	}
}

void nelderMead_calculateVertex(float (*fn)(float*), int &evaluations_used, float &obj, float * p_vertex){
	obj = (*fn)( &p_vertex[0] );
	evaluations_used = evaluations_used + 1;
}

void nelderMead_expansion(int dimension, float expansion_coef, float * p_centroid, float * p_reflection, float * p_expansion){

	for (int i = 0; i < dimension; i++ ) {
		p_expansion[i] = p_centroid[i] + expansion_coef * ( p_reflection[i] - p_centroid[i] );
	}
}

void nelderMead_replacement(int dimension, int index, float * p_simplex, float * p_vertex, float obj, float * p_obj_function){

	for (int i = 0; i < dimension; i++ ) { 
		p_simplex[index * dimension + i] = p_vertex[i]; 
	}
	p_obj_function[index] = obj;
}

void nelderMead_contraction(int dimension, float contraction_coef, float * p_centroid, int index, float * p_simplex, float * p_vertex){

	for (int i = 0; i < dimension; i++ ) {
	 	p_vertex[i] = p_centroid[i] + contraction_coef * ( p_simplex[index * dimension + i] - p_centroid[i] );
	}
}

void nelderMead_shrink(int dimension, int index_best, float * p_simplex){

	for(int i = 0; i < dimension + 1; i++){

		int stride = i * dimension;
		for(int j = 0; j < dimension; j++){
			p_simplex[stride + j] = (p_simplex[stride + j] + p_simplex[index_best * dimension + j]) * 0.5f;
		}
	}
}


float nelmin ( float (*fn)(float*), int dimension, float start[], float xmin[], float reqmin, int konvge, int iterations_number)
{

	float contraction_coef = 0.5f;
	float expansion_coef = 2.0f;
	float reflection_coef = 1.0f;

	float step = 1.0f;

	int index_worst, index_best;

	float best, worst;
	float obj_reflection, obj_vertex;

	int l;
	float x, z;

	
	std::vector<float> simplex(dimension * (dimension + 1)); // p_simplex 
	std::vector<float> centroid(dimension);

	std::vector<float> reflection(dimension);
	std::vector<float> vertex(dimension);

	std::vector<float> obj_function(dimension + 1); // p_obj_function

	float * p_simplex 		 = &simplex[0];
	float * p_centroid 		 = &centroid[0];
	float * p_reflection 	 = &reflection[0];
	float * p_vertex 		 = &vertex[0];
	float * p_obj_function	 = &obj_function[0];

	float * p_start = start;

	int evaluations_used = 0;	

	nelderMead_initialize(dimension, p_simplex, p_start, step);
	nelderMead_calculateSimplex(fn, dimension, evaluations_used, p_simplex, p_obj_function);

	nelderMead_findBest(dimension, best, index_best, p_obj_function);

	//  Inner loop.
	for (int k = 0; k < iterations_number; k++) {

		nelderMead_findWorst(dimension, worst, index_worst, p_obj_function);
		
		//  Calculate centroid, the centroid of the simplex vertices
		//  excepting the vertex with Y value worst.
		nelderMead_centroid(dimension, index_worst, p_simplex, p_centroid);

		//  Reflection through the centroid.
		nelderMead_reflection(dimension, index_worst, reflection_coef, p_simplex, p_centroid, p_reflection);

		nelderMead_calculateVertex(fn, evaluations_used, obj_reflection, p_reflection);
		//  Successful reflection, so extension.
		if ( obj_reflection < best ) {

			nelderMead_expansion(dimension, expansion_coef, p_centroid, p_reflection, p_vertex);
			nelderMead_calculateVertex(fn, evaluations_used, obj_vertex, p_vertex);

		//  Check extension.
			if ( obj_vertex <  obj_reflection) {

				nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
			} else { //  Retain extension or contraction.

				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}
		} else { //  No extension.
			l = 0;
			
			/* ??? */
			for (int i = 0; i < dimension + 1; i++ ) {
				if ( obj_reflection < obj_function[i] ) {
					l += 1;
				}
			}

			if ( 1 < l ) {

				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}
			//  Contraction on the Y(index_worst) side of the centroid.
			else if ( l == 0 ) {

				nelderMead_contraction(dimension, contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);
				nelderMead_calculateVertex(fn, evaluations_used, obj_vertex, p_vertex);

		//  Contract the whole simplex.
				if ( obj_function[index_worst] < obj_vertex ) {

					nelderMead_shrink(dimension, index_best, p_simplex);
					nelderMead_calculateSimplex(fn, dimension, evaluations_used, p_simplex, p_obj_function);

					best = obj_function[0];
					index_best = 0;
				
					for (int i = 1; i < dimension + 1; i++ ) {
						if ( obj_function[i] < best ) { best = obj_function[i]; index_best = i; }
					}
					continue;
				}
		//  Retain contraction.
				else {
					for (int i = 0; i < dimension; i++ ) {
						simplex[i+index_worst* dimension] = vertex[i];
					}
					obj_function[index_worst] = obj_vertex;
				}
			}
		//  Contraction on the reflection side of the centroid.
			else if ( l == 1 ) {
				for (int i = 0; i < dimension; i++ ) {
					vertex[i] = centroid[i] + contraction_coef * ( reflection[i] - centroid[i] );
				}
				obj_vertex = (*fn)( &vertex[0] );
				evaluations_used = evaluations_used + 1;
		//
		//  Retain reflection?
		//
				if ( obj_vertex <= obj_reflection ) {
					for (int i = 0; i < dimension; i++ ) { 
						simplex[i+index_worst* dimension] = vertex[i]; 
					}
					obj_function[index_worst] = obj_vertex;
				}
				else {
					for (int i = 0; i < dimension; i++ ) { 
						simplex[i+index_worst* dimension] = reflection[i]; 
					}
					obj_function[index_worst] = obj_reflection;
				}
			}
		}
		//  Check if best improved.
		if ( obj_function[index_worst] < best ) { 
			best = obj_function[index_worst]; 
			index_best = index_worst; 
		}
		
	}
	return best;
}
#endif

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

void nelderMead_calculateVertex(){
	obj_reflection = (*fn)( &reflection[0] );
		evaluations_used = evaluations_used + 1;
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

		obj_reflection = (*fn)( &reflection[0] );
		evaluations_used = evaluations_used + 1;
		//  Successful reflection, so extension.
		if ( obj_reflection < best ) {
			for (int i = 0; i < dimension; i++ ) {
				vertex[i] = centroid[i] + expansion_coef * ( reflection[i] - centroid[i] );
			}
			obj_vertex = (*fn)( &vertex[0] );
			evaluations_used = evaluations_used + 1;
		//  Check extension.
			if ( obj_reflection < obj_vertex ) {
				for (int i = 0; i < dimension; i++ ) { 
					simplex[i+index_worst* dimension] = reflection[i]; 
				}
				obj_function[index_worst] = obj_reflection;
			} else { //  Retain extension or contraction.
				for (int i = 0; i < dimension; i++ ) { 
					simplex[i+index_worst* dimension] = vertex[i]; 
				}
				obj_function[index_worst] = obj_vertex;
			}
		} else { //  No extension.
			l = 0;
			for (int i = 0; i < dimension + 1; i++ ) {
				if ( obj_reflection < obj_function[i] ) {
					l += 1;
				}
			}

			if ( 1 < l ) {
				for (int i = 0; i < dimension; i++ ) { 
					simplex[i+index_worst* dimension] = reflection[i]; 
				}
				obj_function[index_worst] = obj_reflection;
			}
			//  Contraction on the Y(index_worst) side of the centroid.
			else if ( l == 0 ) {
				for (int i = 0; i < dimension; i++ ) {
					vertex[i] = centroid[i] + contraction_coef * ( simplex[i+index_worst* dimension] - centroid[i] );
				}
				obj_vertex = (*fn)( &vertex[0] );
				evaluations_used = evaluations_used + 1;
		//  Contract the whole simplex.
				if ( obj_function[index_worst] < obj_vertex ) {
					for (int j = 0; j < dimension + 1; j++ ) {
						for (int i = 0; i < dimension; i++ ) {
							simplex[i+j* dimension] = ( simplex[i+j* dimension] + simplex[i+index_best* dimension] ) * 0.5;
							xmin[i] = simplex[i+j* dimension];
						}
						obj_function[j] = (*fn)( xmin );
						evaluations_used = evaluations_used + 1;
					}
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

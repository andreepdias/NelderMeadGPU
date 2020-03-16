#include "util.hpp"

void printVertex(int dimension, float * p_vertex, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", p_vertex[i]);
	}
	printf("\n\n");
}

void printSimplex(int dimension, float * p_simplex, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		int stride = i * dimension;
		printf("%2d. ", i + 1);
		for(int j = 0; j < dimension; j++){
			printf("%.5f ", p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjFunction(int dimension, float * p_obj, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. %.10f\n", i + 1, p_obj[i]);
	}
	printf("\n");
}

void printSingleObjFunction(float x, const char * msg){
	printf("%s: ", msg);
	printf("%.10f\n\n", x);
}

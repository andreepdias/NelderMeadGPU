
void printVertex(int dimension, thrust::device_vector<float> &d_vertex, const char * msg){
	thrust::host_vector<float> h_vertex = d_vertex;

	printf("%s:\n", msg);
	for(int i = 0 ; i < dimension; i++){
		printf("%.5f ", h_vertex[i]);
	}
	printf("\n\n");
}

void printSimplex(int dimension, thrust::device_vector<float> &d_simplex, const char * msg){
	thrust::host_vector<float> h_simplex = d_simplex;

	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. ", i + 1);
		for(int j = 0; j < dimension; j++){
			int stride = i * dimension;
			printf("%.5f ", h_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjFunction(int dimension, thrust::device_vector<float> &d_objective_function, const char * msg){
	thrust::host_vector<float> h_objective_function = d_objective_function;

	printf("%s\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. %.10f\n", i + 1, h_objective_function[i]);
	}
	printf("\n");
}

void printSingleObjFunction(float x, const char * msg){
	printf("%s: ", msg);
	printf("%.10f\n\n", x);

}
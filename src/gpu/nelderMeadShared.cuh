#ifndef NELMINSHARED_H
#define NELMINSHARED_H

#include "util.cuh"
#include "objectiveFunctions.cuh"

void printVertexHost(int dimension, thrust::device_vector<float> &d_vertex, const char * msg, int p = 1){
	thrust::host_vector<float> h_vertex = d_vertex;

	for(int k = 0; k < p; k++){
		int stride = dimension * k;		
		printf("%s [%d]: ", msg, k);
		for(int i = 0 ; i < dimension; i++){
			printf("%.5f ", h_vertex[i + stride]);
		}
		printf("\n");
	}
	printf("\n");
}

void printSimplexHost(int dimension, thrust::device_vector<float> &d_simplex, thrust::device_vector<uint> &d_indexes, const char * msg){
	thrust::host_vector<float> h_simplex = d_simplex;
	thrust::host_vector<uint> h_indexes = d_indexes;

	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. ", h_indexes[i] + 1);
		for(int j = 0; j < dimension; j++){
			int stride = h_indexes[i] * dimension;
			printf("%.5f ", h_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjFunctionHost(int dimension, thrust::device_vector<float> &d_objective_function, thrust::device_vector<uint> &d_indexes, const char * msg){
	thrust::host_vector<float> h_objective_function = d_objective_function;
	thrust::host_vector<uint> h_indexes = d_indexes;

	printf("%s\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. %.10f\n", h_indexes[i] + 1, h_objective_function[i]);
	}
	printf("\n");
}

void printSingleObjFunctionHost(int dimension, thrust::device_vector<float> &d_objective_function, const char * msg, int p = 1){
	thrust::host_vector<float> h_objective_function = d_objective_function;

	printf("%s:\n", msg);
	for(int k = 0; k < p; k++){
		printf("[%d] %.10f\n", k, h_objective_function[k]);
	}
	printf("\n");

}

__device__ void printVertexDevice(int dimension, float * p_vertex, const char * msg, int processor = 0){
	printf("%s [%d]:\n", msg, processor);
	
	int stride = processor * dimension;

	for(int i = 0; i < dimension; i++){
		printf("%.5f ", p_vertex[stride + i]);
	}
	printf("\n\n");
}

__device__ void printSimplexDevice(int dimension, float * p_simplex, uint * p_indexes, const char * msg){
	printf("%s:\n", msg);
	for(int i = 0; i < dimension + 1; i++){
		printf("%2d. ", p_indexes[i] + 1);
		for(int j = 0; j < dimension; j++){
			int stride = p_indexes[i] * dimension;
			printf("%.5f ", p_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

__device__ void printSingleObjFunctionDevice(float * p_obj, const char * msg, int processor = 0){
	printf("%s [%d]:\n", msg, processor);
	printf("%2d. %.10f\n\n", 1, p_obj[processor]);
}

__device__ void printReplacement(const char * msg, int blockId){
	printf("Replacement [%d]: %s.\n", blockId, msg);
}

__device__ void printEvaluationsDevice(int * p_evaluations, int add, int p = 0){
	printf("*-*Total evaluations during update[%d]: %d. Right now adding: %d\n", p, p_evaluations[0], add);
}


void printEvaluationsHost(int total, int add){
	printf("***Total evaluations: %d. Now adding: %d\n", total, add);
}

/* ------------------------- END PRINTING------------------------- */


__global__ void nelderMead_initialize(int dimension, float step, float * start, float * p_simplex){

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

__global__ void nelderMead_centroid(int dimension, float * p_simplex, uint * p_indexes, float * p_centroid, int p = 1){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadsMax = dimension + 1 - p;

	
	int index = p_indexes[threadId];
	int stride = index * dimension;
	
	float value = p_simplex[stride + blockId];
	
	__syncthreads();

	__shared__ float threads_sum [512];
	threads_sum[threadId] = value;
  
	if(threadId < 256 && threadId + 256 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 256];
	}  
	__syncthreads();

	if(threadId < 128 && threadId + 128 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 128];
	}  
	__syncthreads();
	if(threadId < 64 && threadId + 64 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 64];
	}  
	__syncthreads();

	if(threadId < 32 && threadId + 32 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 32];
	}  
	__syncthreads();
  
	if(threadId < 16 && threadId + 16 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 16];
	}  
	__syncthreads();
  
	if(threadId < 8 && threadId + 8 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 8];
	}  
	__syncthreads();
  
	if(threadId < 4 && threadId + 4 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 4];
	}  
	__syncthreads();
  
	if(threadId < 2 && threadId + 2 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 2];
	}  
	__syncthreads();

	if(threadId < 1 && threadId + 1 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 1];
	}  
	__syncthreads();
	
	if(threadId == 0){
		  p_centroid[blockId] = threads_sum[0] / (threadsMax);
	}
}


__device__ void nelderMead_calculate_from_device(int blocks, int dimension, ProblemEnum problem_type, BenchmarkProblemEnum benchmark_problem, void * d_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	if(problem_type == AB_OFF_LATTICE){

		
		ABOffLattice * d_problem_parameters = (ABOffLattice*)d_problem_p;
		int threads = (*d_problem_parameters).protein_length - 2;

		calculateABOffLattice<<< blocks, threads >>>(dimension, d_problem_parameters->protein_length, p_simplex, p_objective_function, is_specific_block, specific_block);
		
	}else if(problem_type == BENCHMARK){

		int threads = dimension;
		
		switch(benchmark_problem){
			case SQUARE:
				calculateSquare<<< blocks, threads >>>(dimension, p_simplex, p_objective_function, is_specific_block, specific_block);
				break;
			case SUM:
				calculateAbsoluteSum<<< blocks, threads >>>(dimension, p_simplex, p_objective_function, is_specific_block, specific_block);
				break;
		}
	}

}

void nelderMead_calculate_from_host(int blocks, NelderMead &p, void * h_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	if(p.problem_type == AB_OFF_LATTICE){

		
		ABOffLattice * h_problem_parameters = (ABOffLattice*)h_problem_p;
		int threads = (*h_problem_parameters).protein_length - 2;

		calculateABOffLattice<<< blocks, threads >>>(p.dimension, h_problem_parameters->protein_length, p_simplex, p_objective_function, is_specific_block, specific_block);
		cudaDeviceSynchronize();
		
	}else if(p.problem_type == BENCHMARK){


		int threads = p.dimension;
		
		switch(p.benchmark_problem){
			case SQUARE:
				calculateSquare<<< blocks, threads >>>(p.dimension, p_simplex, p_objective_function, is_specific_block, specific_block);
				cudaDeviceSynchronize();
				break;
			case SUM:
				calculateAbsoluteSum<<< blocks, threads >>>(p.dimension, p_simplex, p_objective_function, is_specific_block, specific_block);
				cudaDeviceSynchronize();
				break;
		}
	}

}


__global__ void nelderMead_shrink(int dimension, float shrink_coef, float * p_simplex, uint * p_indexes){

    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	
	int stride_a0 = p_indexes[0] * dimension;

    int stride = p_indexes[blockId + 1] * dimension;

	p_simplex[stride +  threadId] = shrink_coef * p_simplex[stride_a0 + threadId] + (1.0f - shrink_coef) * p_simplex[stride + threadId];
}

__device__ void sequence(uint * p_indexes, int end){
	for(int i = 0; i < end; i++){
		p_indexes[i] = i;
	}
}


#endif
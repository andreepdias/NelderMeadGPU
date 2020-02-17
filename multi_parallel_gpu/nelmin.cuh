#ifndef NELMIN_H
#define NELMIN_H

#include "util.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cooperative_groups.h>

__constant__ char aminoacid_sequence[150];


void printStart(int dimension, thrust::device_vector<float> &d_start){
	thrust::host_vector<float> h_start = d_start;

	printf("Start:\n");
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", h_start[i]);
	}
	printf("\n\n");
}

void printInitialize(int dimension, thrust::device_vector<float> &d_simplex){
	thrust::host_vector<float> h_simplex = d_simplex;
	
	printf("Initialize:\n");
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

void printObjectiveFunction(int dimension, thrust::device_vector<float> &d_objective_function, thrust::device_vector<uint> &d_indexes){
	thrust::host_vector<float> h_objective_function = d_objective_function;
	thrust::host_vector<uint> h_indexes = d_indexes;

	printf("Objective Function:\n");
	for(int i = 0; i < dimension + 1; i++){
		printf("%2u. %.10f\n", h_indexes[i] + 1, h_objective_function[i]);
	}
	printf("\n");
}

void printObjectiveFunctionSorted(int dimension, thrust::device_vector<float> &d_objective_function, thrust::device_vector<uint> &d_indexes){
	thrust::host_vector<float> h_objective_function = d_objective_function;
	thrust::host_vector<uint> h_indexes = d_indexes;

	printf("Objective Function Sorted:\n");
	for(int i = 0; i < dimension + 1; i++){
		printf("%2u. %.10f\n", h_indexes[i] + 1, h_objective_function[i]);
	}
	printf("\n");
}

void printCentroid(int dimension, thrust::device_vector<float> &d_centroid){
	thrust::host_vector<float> h_centroid = d_centroid;

	printf("Centroid:\n");
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", h_centroid[i]);
	}
	printf("\n\n");
}

void printReflection(int p, int dimension, thrust::device_vector<float> &d_reflection){
	thrust::host_vector<float> h_reflection = d_reflection;

	for(int k = 0; k < p; k++){
		printf("Reflection [%d]:\n", k);
		for(int i = 0; i < dimension; i++){
			printf("%.5f ", h_reflection[k * dimension + i]);
		}
		printf("\n");
	}
	printf("\n");
}

void printObjectiveFunctionReflection(int p, int dimension, thrust::device_vector<float> &d_objective_function){
	thrust::host_vector<float> h_objective_function = d_objective_function;

	for(int k = 0; k < p; k++){
		printf("Objective Function Reflection [%d]:\n", k);
		printf("%2u. %.10f\n", 1, h_objective_function[0]);
	}
	printf("\n");

}

__device__ void printExpansion(int processor, int dimension, float * p_expansion){

	int stride = processor * dimension;

	printf("Expansion [%d]:\n", processor);
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", p_expansion[stride + i]);
	}
	printf("\n\n");
}

__device__ void printContraction(int processor, int dimension, float * p_contraction){

	int stride = processor * dimension;

	printf("Contraction [%d]:\n", processor);
	for(int i = 0; i < dimension; i++){
		printf("%.5f ", p_contraction[stride + i]);
	}
	printf("\n\n");
}

__device__ void printObjectiveFunctionExpansion(int processor, float * p_obj_expansion){
	printf("Objective Function Expansion [%d]:\n", processor);
	printf("%2d. %.10f\n\n", 1, p_obj_expansion[processor]);
}

__device__ void printObjectiveFunctionContraction(int processor, float * p_obj_contraction){
	printf("Objective Function Contraction [%d]:\n", processor);
	printf("%2d. %.10f\n\n", 1, p_obj_contraction[processor]);
}

__device__ void printReplacement(int blockId, const char * msg){
	printf("Replacement [%d]: %s.\n", blockId, msg);
}

void printSimplex(int dimension, thrust::device_vector<float> &d_simplex, thrust::device_vector<uint> &d_indexes){
	thrust::host_vector<float> h_simplex = d_simplex;
	thrust::host_vector<uint>   h_indexes = d_indexes;
	
	for(int i = 0; i < dimension + 1; i++){
		printf("%2u. ", h_indexes[i] + 1);
		for(int j = 0; j < dimension; j++){
			int stride = h_indexes[i] * dimension;
			printf("%.5f ", h_simplex[stride + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void printPreShrink(int dimension, thrust::device_vector<float> &d_simplex){
	thrust::host_vector<float> h_simplex = d_simplex;

	printf("Pre Shrink:\n");

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

void printShrink(int dimension, thrust::device_vector<float> &d_simplex){
	thrust::host_vector<float> h_simplex = d_simplex;

	printf("Shrink Case 3b (contraction worst than worst vertex):\n");

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

__global__ void nelderMead_initialize(int dimension, float step, float * start, float * p_simplex){

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

__global__ void nelderMead_calculate(int dimension, int protein_length, float * p_simplex, float * p_objective_function, bool is_specific_block = false, int specific_block = 0){

	int blockId = blockIdx.x;

	if(specific_block){
		blockId = specific_block;
	}

	int stride = blockId * dimension;

	int threadId = threadIdx.x;
	int threadsMax = protein_length - 2;
	
	__shared__ float aminoacid_position[150 * 3];

	if(threadId == 0){
		aminoacid_position[0] = 0.0f;
		aminoacid_position[0 + protein_length] = 0.0f;
		aminoacid_position[0 + protein_length * 2] = 0.0f;
	
		aminoacid_position[1] = 0.0f;
		aminoacid_position[1 + protein_length] = 1.0f; 
		aminoacid_position[1 + protein_length * 2] = 0.0f;
	
		aminoacid_position[2] = cosf(p_simplex[stride + 0]);
		aminoacid_position[2 + protein_length] = sinf(p_simplex[stride + 0]) + 1.0f;
		aminoacid_position[2 + protein_length * 2] = 0.0f;
	
		for(int i = 3; i < protein_length; i++){
			aminoacid_position[i] = aminoacid_position[i - 1] + cosf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]); // i - 3 + protein_length - 2
			aminoacid_position[i + protein_length] = aminoacid_position[i - 1 + protein_length] + sinf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]);
			aminoacid_position[i + protein_length * 2] = aminoacid_position[i - 1 + protein_length * 2] + sinf(p_simplex[stride + i + protein_length - 5]);
		}
	}

	__syncthreads();

	float sum = 0.0f, c, d, dx, dy, dz;
	sum += (1.0f - cosf(p_simplex[stride + threadId])) / 4.0f;

	for(unsigned int i = threadId + 2; i < protein_length; i++){

		if(aminoacid_sequence[threadId] == 'A' && aminoacid_sequence[i] == 'A')
			c = 1.0;
		else if(aminoacid_sequence[threadId] == 'B' && aminoacid_sequence[i] == 'B')
			c = 0.5;
		else
			c = -0.5;

		dx = aminoacid_position[threadId] - aminoacid_position[i];
		dy = aminoacid_position[threadId + protein_length] - aminoacid_position[i + protein_length];
		dz = aminoacid_position[threadId + protein_length * 2] - aminoacid_position[i + protein_length * 2];
		d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
		
		sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );		
	}
	__syncthreads();

	__shared__ float threads_sum [128];
	threads_sum[threadId] = sum;
  
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
  
	if(threadId == 0){
		threads_sum[threadId] += threads_sum[threadId + 1];
  
		p_objective_function[blockId] = threads_sum[0];
	}
}

/*
struct Calculate3DAB{
    int dimension;
    int protein_length;

    const float * p_vertex;
    const float * p_aminoacid_position;


    const float * _p_vertex, const float * _p_aminoacid_position, const int _dimension, const int _protein_length)
        : p_vertex(_p_vertex), p_aminoacid_position(_p_aminoacid_position), dimension(_dimension), protein_length(_protein_length)
    {
    };
    
    __device__ float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d, dx, dy, dz;

        sum += (1.0f - cosf(p_vertex[id])) / 4.0f;

        for(unsigned int i = id + 2; i < protein_length; i++){

            if(aminoacid_sequence[id] == 'A' && aminoacid_sequence[i] == 'A')
                c = 1.0;
            else if(aminoacid_sequence[id] == 'B' && aminoacid_sequence[i] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = p_aminoacid_position[id] - p_aminoacid_position[i];
            dy = p_aminoacid_position[id + protein_length] - p_aminoacid_position[i + protein_length];
            dz = p_aminoacid_position[id + protein_length * 2] - p_aminoacid_position[i + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
            
            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
                
        }
        return sum;
    }
};

/* PROBLEM: cant access p_simplex since its a pointer of a device_vector
void nelderMead_calculateSingleVertex(int dimension, int protein_length, float * p_simplex, float * p_objective_function, thrust::device_vector<float> &d_aminoacid_position, thrust::host_vector<float> &h_aminoacid_position) {
	
	h_aminoacid_position[0] = 0.0f;
	h_aminoacid_position[0 + protein_length] = 0.0f;
	h_aminoacid_position[0 + protein_length * 2] = 0.0f;
	
	h_aminoacid_position[1] = 0.0f;
	h_aminoacid_position[1 + protein_length] = 1.0f; 
	h_aminoacid_position[1 + protein_length * 2] = 0.0f;
	
	h_aminoacid_position[2] = cosf(p_simplex[0]);
	h_aminoacid_position[2 + protein_length] = sinf(p_simplex[0]) + 1.0f;
	h_aminoacid_position[2 + protein_length * 2] = 0.0f;
	
	for(int i = 3; i < protein_length; i++){
		h_aminoacid_position[i] = h_aminoacid_position[i - 1] + cosf(p_simplex[i - 2]) * cosf(p_simplex[i + protein_length - 5]); // i - 3 + protein_length - 2
		h_aminoacid_position[i + protein_length] = h_aminoacid_position[i - 1 + protein_length] + sinf(p_simplex[i - 2]) * cosf(p_simplex[i + protein_length - 5]);
		h_aminoacid_position[i + protein_length * 2] = h_aminoacid_position[i - 1 + protein_length * 2] + sinf(p_simplex[i + protein_length - 5]);
	}
	
	d_aminoacid_position = h_aminoacid_position;
	
	float * p_aminoacid_position = thrust::raw_pointer_cast(&d_aminoacid_position[0]);
	
	Calculate3DAB unary_op(p_simplex, p_aminoacid_position, dimension, protein_length);
	
	float result = thrust::transform_reduce(p_simplex, p_simplex + (protein_length - 2), unary_op, 0.0f, thrust::plus<float>());
	
	p_objective_function[0] = result;
}
*/

__global__ void nelderMead_centroid(int dimension, float * p_simplex, uint * p_indexes, float * p_centroid){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadsMax = dimension;

	int index = p_indexes[threadId];
	int stride = index * dimension;

	float value = p_simplex[stride + blockId];

	__syncthreads();

	__shared__ float threads_sum [256];
	threads_sum[threadId] = value;
  
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
	
	if(threadId == 0){
		threads_sum[threadId] += threads_sum[threadId + 1];
  
	  	p_centroid[blockId] = threads_sum[0] / (threadsMax);
	}
}

__global__ void nelderMead_reflection(int p, int dimension, float reflection_coef, float * p_simplex, uint * p_indexes, float * p_centroid, float * p_reflection){

	int index = threadIdx.x;
	int blockId = blockIdx.x;
	int stride = blockId * dimension;

	if(index < dimension){
		p_reflection[stride + index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ p_indexes[dimension - blockId] * dimension + index]);
	}
}

__global__ void nelderMead_expansion(int processor, int dimension, float expansion_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion){

	int index = threadIdx.x;
	int stride = processor * dimension;

	if(index < dimension){
		p_expansion[stride + index] = p_reflection[stride + index] + expansion_coef * (p_reflection[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_contraction(int processor, int dimension, float contraction_coef, float * p_centroid, float * p_vertex, int stride, float * p_contraction){

	int index = threadIdx.x;
	int stride_contraction = processor * dimension;

	if(index < dimension){
		p_contraction[stride_contraction + index] = p_centroid[index] + contraction_coef * (p_vertex[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_replacement(int processor, int dimension, float * p_simplex, float * p_new_vertex, uint * p_indexes, float * p_objective_function, float * p_obj){

	int index = threadIdx.x;

	int stride = processor * dimension;
	int stride_worst = p_indexes[dimension - processor] * dimension;


	if(index < dimension){
		p_simplex[stride_worst + index] = p_new_vertex[stride + index];
	}

	if(threadIdx.x == 0){
		p_objective_function[dimension - processor] = p_obj[processor];
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

__global__ void nelderMead_update(int p, int dimension, int protein_length, float expansion_coef, float contraction_coef, float shrink_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion, float * p_contraction, uint * p_indexes, float * p_objective_function, float * p_obj_reflection, float * p_obj_expansion, float * p_obj_contraction, bool * p_need_shrink){

	int blockId = blockIdx.x;
	float worst = p_objective_function[dimension - blockId];
	float next_worst = p_objective_function[dimension - blockId - 1];

	cooperative_groups::grid_group g = cooperative_groups::this_grid();
	g.sync();

	p_need_shrink[blockId] = false;

	if(p_obj_reflection[blockId] < p_objective_function[0]){

		nelderMead_expansion<<< 1, dimension >>>(blockId, dimension, expansion_coef, p_simplex, p_centroid, p_reflection, p_expansion);
		cudaDeviceSynchronize();
		printExpansion(blockId, dimension, p_expansion);
		
		nelderMead_calculate<<< 1, protein_length - 2 >>>(dimension, protein_length, p_expansion, p_obj_expansion, true, blockId);
		cudaDeviceSynchronize();
		printObjectiveFunctionExpansion(blockId, p_obj_expansion);
		
		if(p_obj_expansion[blockId] < p_objective_function[0]){
			nelderMead_replacement<<< 1, dimension >>>(blockId, dimension, p_simplex, p_expansion, p_indexes, p_objective_function, p_obj_expansion);
			cudaDeviceSynchronize();
			printReplacement(blockId, "Case 1a (expansion better than best vertex)");
		}else{
			nelderMead_replacement<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
			cudaDeviceSynchronize();
			printReplacement(blockId, "Case 1b (reflection better than best vertex)");
		}
	}
	else if(p_obj_reflection[blockId] < next_worst){
		nelderMead_replacement<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
		cudaDeviceSynchronize();
		printReplacement(blockId, "Case 2 (reflection better than second worst vertex)");

	}else{

		bool is_reflection_better = false;

		if(p_obj_reflection[blockId] < worst){
			is_reflection_better = true;

			nelderMead_contraction<<< 1, dimension >>>(blockId, dimension, contraction_coef, p_centroid, p_reflection, blockId * dimension, p_contraction);
			cudaDeviceSynchronize();
		}else{
			nelderMead_contraction<<< 1, dimension >>>(blockId, dimension, contraction_coef, p_centroid, p_simplex, p_indexes[dimension - blockId] * dimension, p_contraction);
			cudaDeviceSynchronize();
		}
		printContraction(blockId, dimension, p_contraction);

		nelderMead_calculate<<< 1, protein_length - 2 >>>(dimension, protein_length, p_contraction, p_obj_contraction, true, blockId);
		cudaDeviceSynchronize();
		printObjectiveFunctionContraction(blockId, p_obj_contraction);
		 

		if(p_obj_contraction[blockId] < worst and p_obj_contraction[blockId] < p_obj_reflection[blockId]){
			nelderMead_replacement<<< 1, dimension >>>(blockId, dimension, p_simplex, p_contraction, p_indexes, p_objective_function, p_obj_contraction);
			cudaDeviceSynchronize();
			printReplacement(blockId, "Case 3a (contraction better than worst vertex)");
		}else{
			p_need_shrink[blockId] = true;

			if(is_reflection_better){
				nelderMead_replacement<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
				cudaDeviceSynchronize();
				printReplacement(blockId, "Case 3b (contraction worse than worst vertex and reflection point -> reflection better than worst)");
			}else{
				printReplacement(blockId, "Case 3c (contraction worse than worst vertex and reflection point -> reflection worse than worst)");
			}

		}
	}
}

void nelderMead(int dim, int psl, float start[], int iterations_number){

	int p = 1;

	int dimension = dim;
	int protein_length = psl;

	float step = 1.0f;
	float reflection_coef = 1.0f;
	float expansion_coef = 1.0f;
	float contraction_coef = 0.5f;
	float shrink_coef = 0.5f;
	
	thrust::device_vector<float> d_start(dimension);	
	
	thrust::device_vector<float> d_aminoacid_position(protein_length * 3);
	thrust::host_vector<float> 	 h_aminoacid_position(protein_length * 3);

	thrust::device_vector<float> d_simplex(dimension * (dimension + 1));	

	thrust::device_vector<float> d_centroid(dimension);
	thrust::device_vector<float> d_reflection(dimension * p);
	thrust::device_vector<float> d_expansion(dimension * p);
	thrust::device_vector<float> d_contraction(dimension * p);
	
    thrust::device_vector<float> d_objective_function(dimension + 1);
	thrust::device_vector<uint>  d_indexes(dimension + 1);

	thrust::device_vector<float> d_obj_reflection(p);	
	thrust::device_vector<float> d_obj_expansion(p);	
	thrust::device_vector<float> d_obj_contraction(p);

	thrust::device_vector<bool>  d_need_shrink(p);

	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
	
	float * p_aminoacid_position 		   = thrust::raw_pointer_cast(&d_aminoacid_position[0]);
	
	float * p_simplex			 		   = thrust::raw_pointer_cast(&d_simplex[0]);
	
	float * p_centroid 			 		   = thrust::raw_pointer_cast(&d_centroid[0]);
	float * p_reflection 		 		   = thrust::raw_pointer_cast(&d_reflection[0]);
	float * p_expansion 		 		   = thrust::raw_pointer_cast(&d_expansion[0]);
	float * p_contraction		 		   = thrust::raw_pointer_cast(&d_contraction[0]);
	
	float * p_objective_function 		   = thrust::raw_pointer_cast(&d_objective_function[0]);
	uint  * p_indexes 				       = thrust::raw_pointer_cast(&d_indexes[0]);
	
	float * p_obj_reflection 			   = thrust::raw_pointer_cast(&d_obj_reflection[0]);
	float * p_obj_expansion 			   = thrust::raw_pointer_cast(&d_obj_expansion[0]);
	float * p_obj_contraction 			   = thrust::raw_pointer_cast(&d_obj_contraction[0]);

	bool * p_need_shrink 				   = thrust::raw_pointer_cast(&d_need_shrink[0]);
	
	thrust::copy(start, start + dimension, d_start.begin());

	printStart(dimension, d_start);


	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, step, p_start, p_simplex);
	cudaDeviceSynchronize();
	printInitialize(dimension, d_simplex);

	thrust::sequence(d_indexes.begin(), d_indexes.end());
	
	nelderMead_calculate<<< dimension + 1, protein_length - 2 >>>(dimension, protein_length, p_simplex, p_objective_function);
	cudaDeviceSynchronize();
	printObjectiveFunction(dimension, d_objective_function, d_indexes);
	
	thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	printObjectiveFunctionSorted(dimension, d_objective_function, d_indexes);
	
	for(int i = 0; i < iterations_number; i++){
		nelderMead_centroid<<< dimension - p + 1, dimension >>>(dimension, p_simplex, p_indexes, p_centroid);
		cudaDeviceSynchronize();
		printCentroid(dimension, d_centroid);
		
		nelderMead_reflection<<< p, dimension >>>(p, dimension, reflection_coef, p_simplex, p_indexes, p_centroid, p_reflection);
		cudaDeviceSynchronize();
		printReflection(p, dimension, d_reflection);
		
		nelderMead_calculate<<< p, protein_length - 2 >>>(dimension, protein_length, p_reflection, p_obj_reflection);
		cudaDeviceSynchronize();
		printObjectiveFunctionReflection(p, dimension, d_obj_reflection);
		
		
		nelderMead_update<<< p, 1 >>>(p, dimension, protein_length, expansion_coef, contraction_coef, shrink_coef, p_simplex, p_centroid, p_reflection, p_expansion, p_contraction, p_indexes, p_objective_function, p_obj_reflection, p_obj_expansion, p_obj_contraction, p_need_shrink);
		cudaDeviceSynchronize();
		printSimplex(dimension, d_simplex, d_indexes);
		
		bool need_shrink = thrust::any_of(d_need_shrink.begin(), d_need_shrink.end(), thrust::identity<bool>());
		
		if(need_shrink){
			printPreShrink(dimension, d_simplex);
			nelderMead_shrink<<< dimension, dimension >>>(dimension, shrink_coef, p_simplex, p_indexes);
			cudaDeviceSynchronize();
			printShrink(dimension, d_simplex);

			thrust::sequence(d_indexes.begin(), d_indexes.end());
			nelderMead_calculate<<< dimension + 1, protein_length - 2 >>>(dimension, protein_length, p_simplex, p_objective_function);
			cudaDeviceSynchronize();
		}
		
		printObjectiveFunction(dimension, d_objective_function, d_indexes);
		thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
		printObjectiveFunctionSorted(dimension, d_objective_function, d_indexes);

		printf("------------------ END ITERATION %d ------------------\n\n", i + 1);
	}
}

#endif
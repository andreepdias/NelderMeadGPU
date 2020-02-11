#ifndef NELMIN_H
#define NELMIN_H

#include "util.h"

__constant__ char aminoacid_sequence[150];


__global__ void nelderMead_initialize(int dimension, float * p_simplex, float step, float * start){

    const int blockId = blockIdx.x;
    const int threadId = threadIdx.x;
    const int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

__global__ void nelderMead_calculate(int protein_length, int dimension, float * p_simplex, float * p_objective_function){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;
	const int threadsMax = protein_length - 2;
	const int stride = blockId * dimension;
	
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

__global__ void nelderMead_centroid(float * p_centroid, float * p_simplex, uint * p_indexes, const int dimension){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;
	const int threadsMax = dimension + 1 - 1;

	const int index = p_indexes[threadId];
	const int stride = index * dimension;

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

__global__ void nelderMead_reflection(float * p_simplex_reflected, float * p_centroid, float * p_simplex, uint * p_indexes, int dimension, float reflection_coef){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;

	const int index = blockId * 32 + threadId;


	if(index < dimension){
		p_simplex_reflected[index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ p_indexes[dimension] * dimension + index]);
	}
}


__global__ void nelderMead_extension(float * p_simplex_extension, float * p_simplex_reflected, float * p_centroid, int dimension, float extension_coef){

	const int blockId = blockIdx.x;
	const int threadId = threadIdx.x;

	const int index = blockId * 32 + threadId;


	if(index < dimension){
		p_simplex_extension[index] = p_simplex_reflected[index] + extension_coef * (p_simplex_reflected[index] - p_centroid[index]);
	}
}


__global__ void nelderMead_update(float * p_simplex_reflected, float * p_simplex_extension, float * p_centroid, float * p_simplex, uint * p_indexes, float * p_objective_function, float * p_objective_function_reflected, float *  p_objective_function_extension, int dimension,  float reflection_coef, float extension_coef){

	const float best = p_objective_function[0];
	const float reflected = p_objective_function_reflected[0];

	if(reflected < best){

		const int numberBlocksExtension = ceil(dimension / 32.0f);

		nelderMead_extension<<< numberBlocksExtension, 32 >>>(p_simplex_extension, p_simplex_reflected, p_centroid, dimension, extension_coef);
		cudaDeviceSynchronize();
	}
	
	// else{
	// 	const int next_index = (index + 1) % p;

	// 	if(p_objective_function_reflected[index] < p_objective_function[next_index + p]){
	// 		nelderMead_updateReplace<<< >>>();
	// 		cudaDeviceSynchronize();

	// 	}else{
	// 		const bool use_reflected = false;

	// 		if(p_objective_function_reflected[index] < p_objective_function[index + p]){
	// 			use_reflected = true;
	// 		}
	// 		nelderMead_updateContract<<< >>>();
	// 		cudaDeviceSynchronize();

	// 	}
	// }
}

void nelderMead(int dimension, int protein_length, float start[]){

	const int n = dimension;
	const int psl = protein_length;
	
	const float step = 1.0f;
	const float reflection_coef = 1.0f;
	const float extension_coef = 1.0f;

    thrust::device_vector<float> d_objective_function(n + 1);
	thrust::device_vector<uint>  d_indexes(n + 1);
	thrust::device_vector<float> d_start(n);
	thrust::device_vector<float> d_simplex(n * (n + 1));
	thrust::device_vector<float> d_centroid(n);
	
	thrust::device_vector<float> d_simplex_reflected(n);
	thrust::device_vector<float> d_objective_function_reflected(1);	
	thrust::device_vector<float> d_simplex_extension(n);
    thrust::device_vector<float> d_objective_function_extension(1);


	float * p_objective_function 		   = thrust::raw_pointer_cast(&d_objective_function[0]);
	uint  * p_indexes 				       = thrust::raw_pointer_cast(&d_indexes[0]);
	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
	float * p_simplex			 		   = thrust::raw_pointer_cast(&d_simplex[0]);
	float * p_centroid 			 		   = thrust::raw_pointer_cast(&d_centroid[0]);
	
	float * p_simplex_reflected	 		   = thrust::raw_pointer_cast(&d_simplex_reflected[0]);
	float * p_objective_function_reflected = thrust::raw_pointer_cast(&d_objective_function_reflected[0]);
	float * p_simplex_extension	 		   = thrust::raw_pointer_cast(&d_simplex_extension[0]);
	float * p_objective_function_extension = thrust::raw_pointer_cast(&d_objective_function_extension[0]);

	thrust::copy(start, start + n, d_start.begin());

	nelderMead_initialize<<< n + 1, n >>>(dimension, p_simplex, step, p_start);

	nelderMead_calculate<<< n + 1, psl - 2 >>>(psl, n, p_simplex, p_objective_function);
	
	thrust::sequence(d_indexes.begin(), d_indexes.end());
	thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	
	nelderMead_centroid<<< n, n + 1 - 1 >>>(p_centroid, p_simplex, p_indexes, dimension);
	
	int numberBlocksReflection = ceil(n / 32.0f);
	
	nelderMead_reflection<<< numberBlocksReflection, 32 >>>(p_simplex_reflected, p_centroid, p_simplex, p_indexes, dimension, reflection_coef);
	
	nelderMead_calculate<<< 1, psl - 2 >>>(psl, n, p_simplex_reflected, p_objective_function_reflected);

	nelderMead_update<<< 1, 1 >>> (p_simplex_reflected, p_simplex_extension, p_centroid, p_simplex, p_indexes, p_objective_function, p_objective_function_reflected, p_objective_function_extension, dimension, reflection_coef, extension_coef );
	
	/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */

	thrust::host_vector<float> h_simplex = d_simplex;
	thrust::host_vector<float> h_centroid = d_centroid;
	thrust::host_vector<float> h_objective_function = d_objective_function;
	thrust::host_vector<float> h_indexes = d_indexes;
	thrust::host_vector<float> h_simplex_reflected = d_simplex_reflected;
	thrust::host_vector<float> h_objective_function_reflected = d_objective_function_reflected;

	printf("n: %d\tpsl: %d\n", n, psl);

	for(int i  = 0; i <= n; i++){
		printf("i: %d\tobj: %5.5f\t", i + 1, h_objective_function[i]);
		for(int j = 0; j < n; j++){
			printf("%5.5f ", h_simplex[h_indexes[i] *  dimension + j]);
		}
		printf("\n");
	}
	printf("\n");
	for(int j = 0; j < n; j++){
		printf("%5.5f ", h_centroid[j]);
	}
	printf("\n\n");

	for(int j = 0; j < n; j++){
		printf("%5.5f ", h_simplex_reflected[j]);
	}
	printf("\n\n");

	printf("%5.5f\n", h_objective_function_reflected[0]);
	
	// nelderMead_calculate<<< p, psl - 2 >>>(psl, n, p_simplex_reflected, p_objective_function_reflected);
	
	// nelderMead_update<<< p, 1 >>>(p_simplex_reflected, p_centroid, p_simplex, p_indexes, p_objective_function, p_objective_function_reflected, dimension, p, reflection_coef);




}



#endif
#pragma once 

#include "../shared/objectiveFunctions.cuh"
#include "../shared/util.cuh"

__global__ void nelderMead_initialize(const int dimension, const float step, const float * __restrict__ p_start, float * p_simplex){

    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int index = threadId * dimension + blockId;
	
	__shared__ float s;
	float l;

	if(threadId == 0){
		s = p_start[blockId];
	}
	__syncthreads();

	l = s;	
	
	if(threadId == blockId){
		l += step;
	}
	__syncthreads();

	p_simplex[index] = l;
}

__global__ void nelderMead_centroid(const int dimension, const float * __restrict__ p_simplex, const uint * __restrict__ p_indexes, float * p_centroid, const int p = 1){

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


__global__ void nelderMead_shrink(const int dimension, const float shrink_coef, float * p_simplex, const uint * __restrict__ p_indexes){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	
	__shared__ float best;

	int stride = p_indexes[threadId + 1] * dimension;
	
	if(threadId == 0){
		best = p_simplex[p_indexes[0] * dimension + blockId];
	}
	__syncthreads();

	p_simplex[stride +  blockId] = shrink_coef * best + (1.0f - shrink_coef) * p_simplex[stride + blockId];
}
__device__ void sequence(uint * p_indexes, int end){
	for(int i = 0; i < end; i++){
		p_indexes[i] = i;
	}
}

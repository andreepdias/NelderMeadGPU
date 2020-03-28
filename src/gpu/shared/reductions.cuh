#pragma once

#include "util.cuh"

__global__ void countIf(int * count, const float * __restrict__ p_obj_function, const int dimension, const float obj_reflection) {
  
	__shared__ int sharedInc;

	float obj;
	int index =  threadIdx.x + blockIdx.x * blockDim.x;

	if (threadIdx.x == 0){
		sharedInc = 0;
	}
	__syncthreads();

	if(index < dimension) {
		obj = p_obj_function[index];
		
		if(obj_reflection < obj){
			atomicAdd(&sharedInc, 1);
		}
	}
	__syncthreads();

	if(threadIdx.x == 0){
		atomicAdd(count, sharedInc);
	}
}

__device__ void atomicMax(float * const address, const float value)
{
	if (*address >= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI;
	int assumed;

	do 
	{
		assumed = old;

		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
	} while (assumed != old);
}

__global__ void findMax(const float* __restrict__ input, const int size, float * out, int * outIdx)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ float threads_max[32];
	__shared__ int   threads_id[32];
	
	int threadId = threadIdx.x;

	if(index < size){
		
		threads_max[threadId] = input[index];
		threads_id[threadId] = index;
	  
		__syncthreads();
	
	  
		if(threadId < 16 and threadId + 16 < 32 and index + 16 < size){
			if(threads_max[threadId] < threads_max[threadId + 16]){
				threads_max[threadId] = threads_max[threadId + 16];
				threads_id[threadId] = threads_id[threadId + 16];
			}
		}  
		__syncthreads();
	  
		if(threadId < 8 and threadId + 8 < 32 and index + 8 < size){
			if(threads_max[threadId] < threads_max[threadId + 8]){
				threads_max[threadId] = threads_max[threadId + 8];
				threads_id[threadId] = threads_id[threadId + 8];
			}
		}  
		__syncthreads();
	  
		if(threadId < 4 and threadId + 4 < 32 and index + 4 < size){
			if(threads_max[threadId] < threads_max[threadId + 4]){
				threads_max[threadId] = threads_max[threadId + 4];
				threads_id[threadId] = threads_id[threadId + 4];
			}
		}  
		__syncthreads();
	  
		if(threadId < 2 and threadId + 2 < 32 and index + 2 < size){
			if(threads_max[threadId] < threads_max[threadId + 2]){
				threads_max[threadId] = threads_max[threadId + 2];
				threads_id[threadId] = threads_id[threadId + 2];
			}
		}  
		__syncthreads();
	
		if(threadId < 1 and threadId + 1 < 32 and index + 1 < size){
			if(threads_max[threadId] < threads_max[threadId + 1]){
				threads_max[threadId] = threads_max[threadId + 1];
				threads_id[threadId] = threads_id[threadId + 1];
			}		
		}
		__syncthreads();
		
		if (threadIdx.x == 0)
		{
			atomicMax(out, threads_max[0]);

			cooperative_groups::grid_group g = cooperative_groups::this_grid();
			g.sync();

			if(*out == threads_max[0]){
				*outIdx = threads_id[0];
			}
		}
	}
}

__device__ void atomicMin(const float * address, const float value)
{
	if (*address <= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI;
	int assumed;

	do 
	{
		assumed = old;

		if (__int_as_float(assumed) <= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
	} while (assumed != old);
}


__global__ void findMin(const float * __restrict__ input, const int size, float * out, int * outIdx)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ float threads_min[32];
	__shared__ int   threads_id[32];
	
	int threadId = threadIdx.x;

	if(index < size){
		
		threads_min[threadId] = input[index];
		threads_id[threadId] = index;
	  
		__syncthreads();
	
	  
		if(threadId < 16 && threadId + 16 < 32){
			if(threads_min[threadId] > threads_min[threadId + 16]){
				threads_min[threadId] = threads_min[threadId + 16];
				threads_id[threadId] = threads_id[threadId + 16];
			}
		}  
		__syncthreads();
	  
		if(threadId < 8 && threadId + 8 < 32){
			if(threads_min[threadId] > threads_min[threadId + 8]){
				threads_min[threadId] = threads_min[threadId + 8];
				threads_id[threadId] = threads_id[threadId + 8];
			}
		}  
		__syncthreads();
	  
		if(threadId < 4 && threadId + 4 < 32){
			if(threads_min[threadId] > threads_min[threadId + 4]){
				threads_min[threadId] = threads_min[threadId + 4];
				threads_id[threadId] = threads_id[threadId + 4];
			}
		}  
		__syncthreads();
	  
		if(threadId < 2 && threadId + 2 < 32){
			if(threads_min[threadId] > threads_min[threadId + 2]){
				threads_min[threadId] = threads_min[threadId + 2];
				threads_id[threadId] = threads_id[threadId + 2];
			}
		}  
		__syncthreads();
	
		if(threadId < 1 && threadId + 1 < 32){
			if(threads_min[threadId] > threads_min[threadId + 1]){
				threads_min[threadId] = threads_min[threadId + 1];
				threads_id[threadId] = threads_id[threadId + 1];
			}		
		}
		__syncthreads();
		
		if (threadIdx.x == 0)
		{
			atomicMin(out, threads_min[0]);

			cooperative_groups::grid_group g = cooperative_groups::this_grid();
			g.sync();

			if(*out == threads_min[0]){
				*outIdx = threads_id[0];
			}
		}
	}
}

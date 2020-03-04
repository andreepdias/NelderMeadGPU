#include "util.cuh"

__global__ void calculateSquare(int dimension, float * p_simplex, float * p_objective_function, bool is_specific_block = false, int specific_block = 0){

    int blockId = blockIdx.x;
    
    if(specific_block){
		blockId = specific_block;
    }
    
    int stride = blockId * dimension;
    
	int threadId = threadIdx.x;
    int threadsMax = dimension;
    
    float square = (p_simplex[stride + threadId] * p_simplex[stride + threadId]) / 100.0f;
    
    __syncthreads();

    __shared__ float threads_sum [256];
    threads_sum[threadId] = square;
    
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
  
		p_objective_function[blockId] = threads_sum[0];
	}
}

__global__ void calculateAbsoluteSum(int dimension, float * p_simplex, float * p_objective_function, bool is_specific_block = false, int specific_block = 0){

    int blockId = blockIdx.x;
    
    if(specific_block){
		blockId = specific_block;
    }
    
    int stride = blockId * dimension;
    
	int threadId = threadIdx.x;
    int threadsMax = dimension;
    
    float absolute = abs(p_simplex[stride + threadId]) / 100.0f;
    
    __syncthreads();

    __shared__ float threads_sum [256];
    threads_sum[threadId] = absolute;
    
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
  
		p_objective_function[blockId] = threads_sum[0];
	}
}

__global__ void calculateABOffLattice(int dimension, int protein_length, float * p_simplex, float * p_objective_function, bool is_specific_block = false, int specific_block = 0){

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
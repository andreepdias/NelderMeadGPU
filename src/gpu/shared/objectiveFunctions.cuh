#pragma once

#include "util.cuh"

__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__global__ void calculateSingleAB3D(float * obj, const int protein_length, const float * __restrict__ vertex){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	// int threadsMax = protein_length - 2;
	int index = blockId + threadId + 2;

	float sum = 0.0f, c, d, x, y;
	char aa_thread, aa_block;
	float3 pos_thread, pos_block, subtract;

	__shared__ float3 aminoacid_position[150];

	if(threadId == 0){
		aminoacid_position[0].x = 0.0f;
		aminoacid_position[0].y = 0.0f;
		aminoacid_position[0].z = 0.0f;
	
		aminoacid_position[1].x = 0.0f;
		aminoacid_position[1].y = 1.0f; 
		aminoacid_position[1].z = 0.0f;
	
		x = vertex[0];

		aminoacid_position[2].x = cosf(x);
		aminoacid_position[2].y = sinf(x) + 1.0f;
		aminoacid_position[2].z = 0.0f;
	
		for(int i = 3; i < protein_length; i++){

			x = vertex[i - 2];
			y = vertex[i + protein_length - 5];

			pos_thread = aminoacid_position[i - 1];

			aminoacid_position[i].x = pos_thread.x + cosf(x) * cosf(y); // i - 3 + protein_length - 2
			aminoacid_position[i].y = pos_thread.y + sinf(x) * cosf(y);
			aminoacid_position[i].z = pos_thread.z + sinf(y);
		}
		
		sum += (1.0f - cosf(vertex[blockId])) / 4.0f;
	}

	__syncthreads();
	
	if(index < protein_length){

		aa_block = aminoacid_sequence[blockId];
		aa_thread = aminoacid_sequence[index];
	
		if(aa_block == 'A' and aa_thread == 'A'){
			c = 1.0;
		}else if(aa_block == 'B' and aa_thread == 'B'){
			c = 0.5;
		}else{
			c = -0.5;
		}
		__syncthreads();
		
		pos_thread = aminoacid_position[index];
		pos_block = aminoacid_position[blockId];
	
		subtract = pos_thread - pos_block;
	
		d = norm3df(subtract.x, subtract.y, subtract.z);
		sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );		
	
		__syncthreads();
	
		__shared__ float threads_sum [256];
		threads_sum[threadId] = sum;
	
		if(threadId < 128 && index + 128 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 64];
		}  
		__syncthreads();
	  
		if(threadId < 64 && index + 64 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 64];
		}
		__syncthreads();
		
		if(threadId < 32 && index + 32 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 32];
		}
		__syncthreads();
		
		if(threadId < 16 && index + 16 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 16];
		}  
		__syncthreads();
	  
		if(threadId < 8 && index + 8 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 8];
		}  
		__syncthreads();
	  
		if(threadId < 4 && index + 4 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 4];
		}  
		__syncthreads();
	  
		if(threadId < 2 && index + 2 < protein_length){
			threads_sum[threadId] += threads_sum[threadId + 2];
		}  
		__syncthreads();
	  
		if(threadId == 0){
			if(index + 1 < protein_length){
				threads_sum[threadId] += threads_sum[threadId + 1];
			}
	
			atomicAdd(obj, threads_sum[0]);
		}
	}
}


__global__ void calculateABOffLattice(const int dimension, const int protein_length, const float * __restrict__ p_simplex, float * p_objective_function, bool is_specific_block = false, int specific_block = 0){

	int blockId = blockIdx.x;

	if(is_specific_block){
		blockId = specific_block;
	}

	int threadId = threadIdx.x;

    int stride = blockId * dimension;
	int threadsMax = protein_length - 2;

	float sum = 0.0f, c, d, x, y;
	char aa_thread, aa_i;
	float3 pos_i, pos_thread, subtract;
	
	__shared__ float3 aminoacid_position[150];

	if(threadId == 0){
		aminoacid_position[0].x = 0.0f;
		aminoacid_position[0].y = 0.0f;
		aminoacid_position[0].z = 0.0f;
	
		aminoacid_position[1].x = 0.0f;
		aminoacid_position[1].y = 1.0f; 
		aminoacid_position[1].z = 0.0f;
	
		x = p_simplex[stride + 0];

		aminoacid_position[2].x = cosf(x);
		aminoacid_position[2].y = sinf(x) + 1.0f;
		aminoacid_position[2].z = 0.0f;
	
		for(int i = 3; i < protein_length; i++){

			x = p_simplex[stride + i - 2];
			y = p_simplex[stride + i + protein_length - 5];

			pos_i = aminoacid_position[i - 1];

			aminoacid_position[i].x = pos_i.x + cosf(x) * cosf(y); // i - 3 + protein_length - 2
			aminoacid_position[i].y = pos_i.y + sinf(x) * cosf(y);
			aminoacid_position[i].z = pos_i.z + sinf(y);
		}
	}

	__syncthreads();

	sum += (1.0f - cosf(p_simplex[stride + threadId])) / 4.0f;

	for(unsigned int i = threadId + 2; i < protein_length; i++){

		aa_thread = aminoacid_sequence[threadId];
		aa_i = aminoacid_sequence[i];

		if(aa_thread == 'A' && aa_i == 'A')
			c = 1.0f;
		else if(aa_thread == 'B' && aa_i == 'B')
			c = 0.5f;
		else
			c = -0.5f;

		pos_thread = aminoacid_position[threadId];
		pos_i = aminoacid_position[i];

		subtract = pos_thread - pos_i;

		d = norm3df(subtract.x, subtract.y, subtract.z);
		
		sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );		
	}
	__syncthreads();

	__shared__ float threads_sum [256];
	threads_sum[threadId] = sum;

	if(threadId < 128 && threadId + 128 < threadsMax){
		threads_sum[threadId] += threads_sum[threadId + 64];
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

void calculateSingle3DABOffLattice(const int dimension, const float * p_vertex, const void * problem_parameters, float * obj){
	
	ABOffLattice * parametersAB = (ABOffLattice*)problem_parameters;
	int protein_length = (*parametersAB).protein_length;

	// *obj = 0;
	// calculateSingleAB3D<<<protein_length - 2, protein_length - 2 >>>(obj, protein_length, p_vertex);
	calculateABOffLattice<<< 1, protein_length - 2 >>>(dimension, protein_length, p_vertex, obj);
	cudaDeviceSynchronize();
}

void calculateMulti3DABOffLattice(const int blocks, const int dimension, const float * p_simplex, float * p_obj_function, const void * problem_parameters){
	
	ABOffLattice * parametersAB = (ABOffLattice*)problem_parameters;
	int protein_length = (*parametersAB).protein_length;

	calculateABOffLattice<<< blocks, protein_length - 2 >>>(dimension, protein_length, p_simplex, p_obj_function);
	cudaDeviceSynchronize();
}



__device__ void nelderMead_calculateFromDevice(int blocks, int dimension, ProblemEnum problem_type, BenchmarkProblemEnum benchmark_problem, void * d_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	ABOffLattice * d_problem_parameters = (ABOffLattice*)d_problem_p;
	int threads = (*d_problem_parameters).protein_length - 2;

	calculateABOffLattice<<< blocks, threads >>>(dimension, d_problem_parameters->protein_length, p_simplex, p_objective_function, is_specific_block, specific_block);
	cudaDeviceSynchronize();
}

void nelderMead_calculateFromHost(int blocks, NelderMead &p, void * h_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	ABOffLattice * h_problem_parameters = (ABOffLattice*)h_problem_p;
	int threads = (*h_problem_parameters).protein_length - 2;

	calculateABOffLattice<<< blocks, threads >>>(p.dimension, h_problem_parameters->protein_length, p_simplex, p_objective_function);
	cudaDeviceSynchronize();

}

__device__ void nelderMead_calculateSingleFromDevice(void * d_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	ABOffLattice * d_problem_parameters = (ABOffLattice*)d_problem_p;
	int protein_length = (*d_problem_parameters).protein_length ;

	p_objective_function[0] = 0;
	calculateSingleAB3D<<< protein_length - 2, protein_length - 2 >>>(&p_objective_function[0], protein_length, p_simplex);
	cudaDeviceSynchronize();
}

void nelderMead_calculateSingleFromHost(void * h_problem_p, float * p_simplex, float * p_objective_function,  bool is_specific_block = false, int specific_block = 0){

	ABOffLattice * h_problem_parameters = (ABOffLattice*)h_problem_p;
	int protein_length = (*h_problem_parameters).protein_length;

	calculateSingleAB3D<<< protein_length - 2, protein_length - 2 >>>(&p_objective_function[0], protein_length, p_simplex);
	cudaDeviceSynchronize();

}



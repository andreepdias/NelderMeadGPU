#pragma once

#include "../shared/reductions.cuh"
#include "../shared/objectiveFunctions.cuh"
#include "../shared/util.cuh"
#include "../shared/print.cuh"


void nelderMead_calculateVertexFast(const int dimension, int &evaluations_used, float &h_obj, const float * p_vertex, const void * problem_parameters, float * obj){
	
	nelderMead_calculateFromHost(1, dimension, problem_parameters, p_vertex, obj);
	evaluations_used += 1;

	h_obj = *obj;
}


void nelderMead_calculateSimplexFast(const int blocks, const int dimension, int &evaluations_used, float * p_obj_function, const float * p_simplex, const void * problem_parameters){

	nelderMead_calculateFromHost(blocks, dimension, problem_parameters, p_simplex, p_obj_function);
	evaluations_used += blocks;
}

void nelderMead_findBestFast(const int dimension, const int numberBlocks,float &best, int &index_best, const float * p_obj_function, float * obj, int * idx){

	findMin <<< numberBlocks, 32 >>>(p_obj_function, dimension + 1, obj, idx);
	cudaDeviceSynchronize();
	
	best = *obj;
	index_best = *idx;
}

void nelderMead_findWorstFast(const int dimension, const int numberBlocks, float &worst, int &index_worst, const float * p_obj_function, float * obj, int * idx){

	findMax <<< numberBlocks, 32 >>>(p_obj_function, dimension + 1, obj, idx);
	cudaDeviceSynchronize();
	
	worst = *obj;
	index_worst = *idx;
}


__global__ void nelderMead_initializeFast(const int dimension, const float step, const float * __restrict__ p_start, float * p_simplex){

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

__global__ void nelderMead_centroidFast(const int dimension, const int index_worst, const float * __restrict__ p_simplex, float * p_centroid){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadsMax = dimension + 1;

	__shared__ float worst;

	float value = p_simplex[threadId * dimension + blockId];

	if(threadId == index_worst){
		worst = value;
	}
	
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
		p_centroid[blockId] = (threads_sum[0] - worst) / (dimension);
	}
}

__global__ void nelderMead_reflectionFast(const int dimension, const float reflection_coef, const float * __restrict__ p_simplex, const int index_worst, const float * __restrict__ p_centroid, float * p_reflection){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	float c = p_centroid[index];

	if(index < dimension){
		p_reflection[index] = c + reflection_coef * (c - p_simplex[ index_worst * dimension + index]);
	}
}

__global__ void nelderMead_expansionFast(const int dimension, const float expansion_coef, const float * __restrict__ p_centroid, const float * __restrict__ p_reflection, float * p_expansion){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	float c = p_centroid[index];

	if(index < dimension){
		p_expansion[index] = c + expansion_coef * (p_reflection[index] - c);
	}
}

__global__ void nelderMead_replacementFast(const int dimension, float * p_simplex, const float * __restrict__ p_new_vertex, const int index_worst, float * p_obj_function, const float obj, float * recent){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 
	int stride = index_worst * dimension;


	if(index < dimension){
		p_simplex[stride + index] = p_new_vertex[index];
	}

	if(blockId == 0 and threadId == 0){
		p_obj_function[index_worst] = obj;
		*recent = obj;
	}
}

__global__ void nelderMead_contractionFast(const int dimension, const float contraction_coef, const float * __restrict__ p_centroid, int index, const float * __restrict__ p_simplex, float * p_vertex){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int i = blockId * 32 + threadId; 

	float c = p_centroid[i];

	if(i < dimension){
		p_vertex[i] = c + contraction_coef * (p_simplex[index * dimension + i] - c);
	}
}

__global__ void nelderMead_shrinkFast(const int dimension, const float shrink_coef, float * p_simplex, const int index_best){

    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	
	int stride = threadId * dimension;
	
	__shared__ float best;

	if(threadId == 0){
		best = p_simplex[index_best * dimension + blockId];
	}
	__syncthreads();

	if(threadId != index_best){
		p_simplex[stride +  blockId] = shrink_coef * best + (1.0f - shrink_coef) * p_simplex[stride + blockId];
	}
}

NelderMeadResult nelderMeadFast (NelderMead &parameters, std::ofstream &outputFile, void * problem_parameters = NULL)
{

	int dimension = parameters.dimension;

	parameters.step = 1.0f;

	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 2.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;
	
	/*
	parameters.reflection_coef = 1.0f; //alpha
	parameters.expansion_coef = 1.0f + (2.0f / dimension); // beta
	parameters.contraction_coef = 0.75f - (1.0f / (2.0f *  dimension)); // gama
	parameters.shrink_coef = 1.0f - (1.0f / dimension); // omega
	*/

	thrust::device_vector<float> d_start(dimension);	
	thrust::device_vector<float> d_simplex(dimension * (dimension + 1));
	thrust::device_vector<float> d_centroid(dimension);
	thrust::device_vector<float> d_reflection(dimension);
	thrust::device_vector<float> d_vertex(dimension);
	thrust::device_vector<float> d_obj_function(dimension + 1);
	
	float best, worst, obj_reflection, obj_vertex;
	int index_best, index_worst;

	float * p_start 		= thrust::raw_pointer_cast(&d_start[0]);
	float * p_simplex 		= thrust::raw_pointer_cast(&d_simplex[0]);
	float * p_centroid 		= thrust::raw_pointer_cast(&d_centroid[0]);
	float * p_reflection 	= thrust::raw_pointer_cast(&d_reflection[0]);
	float * p_vertex 		= thrust::raw_pointer_cast(&d_vertex[0]);
	float * p_obj_function 	= thrust::raw_pointer_cast(&d_obj_function[0]);

	float * obj;
	int * idx;
	int * count;
	float * recent;

	int eval = 0;
	
	cudaMallocManaged(&obj, sizeof(float));
	cudaMallocManaged(&idx, sizeof(int));
	cudaMallocManaged(&count, sizeof(int));
	cudaMallocManaged(&recent, sizeof(float));

	cudaMemset(obj, 0.0f, sizeof(float));	
	cudaMemset(idx, 0, sizeof(int));
	cudaMemset(count, 0, sizeof(int));
	cudaMemset(recent, 0.0f, sizeof(float));	

	int numberBlocks = ceil((float)dimension / 32.0f);


	thrust::copy(parameters.p_start, parameters.p_start + dimension, d_start.begin());	
	
	nelderMead_initializeFast<<< dimension, dimension + 1 >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();

	nelderMead_calculateSimplexFast(dimension + 1, dimension, eval, p_obj_function, p_simplex, problem_parameters);

	*idx = index_best = index_worst = 0;
	*recent = *obj = best = worst = d_obj_function[0];

	nelderMead_findBestFast(dimension, numberBlocks, best, index_best, p_obj_function, obj, idx);

	printf("[%d] (%d): %.7f\n", eval, parameters.evaluations_number, best);
	for (;eval < parameters.evaluations_number;) {
		
		*obj = best;
		// *idx = index_best;
		nelderMead_findWorstFast(dimension, numberBlocks, worst, index_worst, p_obj_function, obj, idx);
		
		nelderMead_centroidFast<<< dimension, dimension + 1>>>(dimension, index_worst, p_simplex, p_centroid);
		cudaDeviceSynchronize();
		
		nelderMead_reflectionFast<<< numberBlocks, 32 >>>(dimension, parameters.reflection_coef, p_simplex, index_worst, p_centroid, p_reflection);
		cudaDeviceSynchronize();

		nelderMead_calculateVertexFast(dimension, eval, obj_reflection, p_reflection, problem_parameters, obj);
		
		if(obj_reflection < best){
			nelderMead_expansionFast<<< numberBlocks, 32 >>>(dimension, parameters.expansion_coef, p_centroid, p_reflection, p_vertex);
			cudaDeviceSynchronize();
			
			nelderMead_calculateVertexFast(dimension, eval, obj_vertex, p_vertex, problem_parameters, obj);

			if(obj_vertex < best){

				nelderMead_replacementFast<<< numberBlocks, 32 >>>(dimension, p_simplex, p_vertex, index_worst, p_obj_function, obj_vertex, recent);
				cudaDeviceSynchronize();
				worst = *recent;
			}else{
				
				nelderMead_replacementFast<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection, recent);
				cudaDeviceSynchronize();
				worst = *recent;
			}
		}else{
			*count = 0;
			countIf<<< numberBlocks, 32 >>>(count, p_obj_function, dimension, obj_reflection);
			cudaDeviceSynchronize();
			
			/* Se reflection melhor que segundo pior vértice (e pior) */
			if(*count >= 2){
				
				nelderMead_replacementFast<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection, recent);
				cudaDeviceSynchronize();
				worst = *recent;
			}else{
				if(obj_reflection < worst){
					
					nelderMead_contractionFast<<< numberBlocks, 32 >>>(dimension, parameters.contraction_coef, p_centroid, 0, p_reflection, p_vertex);
					cudaDeviceSynchronize();
				}else{
					nelderMead_contractionFast<<< numberBlocks, 32 >>>(dimension, parameters.contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);
					cudaDeviceSynchronize();
				}
				nelderMead_calculateVertexFast(dimension, eval, obj_vertex, p_vertex, problem_parameters, obj);
				
				if(obj_vertex < obj_reflection and obj_vertex < worst){
					
					nelderMead_replacementFast<<< numberBlocks, 32 >>>(dimension, p_simplex, p_vertex, index_worst, p_obj_function, obj_vertex, recent);
					cudaDeviceSynchronize();
					worst = *recent;
					
				}else if(obj_reflection < worst){
					
					nelderMead_replacementFast<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection, recent);
					cudaDeviceSynchronize();
					worst = *recent;
				}else{
					nelderMead_shrinkFast<<< dimension, dimension + 1 >>>(dimension, parameters.shrink_coef, p_simplex, index_best);

					cudaDeviceSynchronize();
					nelderMead_calculateSimplexFast(dimension + 1, dimension, eval, p_obj_function, p_simplex, problem_parameters);
					
					*obj = best;
					*idx = index_best;
					nelderMead_findBestFast(dimension, numberBlocks, best, index_best, p_obj_function, obj, idx);
					nelderMead_findWorstFast(dimension, numberBlocks, worst, index_worst, p_obj_function, obj, idx);

				}
			}
		}
		if (worst < best){ 
			best = worst; 
			index_best = index_worst; 			
			printf("[%d] (%d): %.7f\n", eval, parameters.evaluations_number, best);
		}
	}
	// outputFile << eval << ' ' << best << std::endl;
	
	NelderMeadResult result;
	
	result.best = best;
	result.best_vertex.resize(dimension);
	result.evaluations_used = eval;
	
	int stride = index_best * dimension;
	thrust::copy(d_simplex.begin() + stride, d_simplex.begin() + stride + dimension, result.best_vertex.begin());
	
	// printf("bbb\n");
	return result;
}

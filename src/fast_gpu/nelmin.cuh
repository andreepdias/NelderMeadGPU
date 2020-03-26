#ifndef NELMIN_H
#define NELMIN_H

#include "util.cuh"
#include "print.cuh"
#include "objectiveFunctions.cuh"


// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html


__global__ void nelderMead_initialize(int dimension, float step, float * start, float * p_simplex){

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

void nelderMead_calculateVertex(int dimension, int &evaluations_used, float &h_obj, float * p_vertex, void * problem_parameters, float * obj){
	
	calculate3DABOffLattice(dimension, p_vertex, problem_parameters, obj);
	evaluations_used = evaluations_used + 1;

	h_obj = *obj;
}


__device__ void atomicMax(float* const address, const float value)
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
	
	  
		if(threadId < 16 && threadId + 16 < 32){
			if(threads_max[threadId] < threads_max[threadId + 16]){
				threads_max[threadId] = threads_max[threadId + 16];
				threads_id[threadId] = threads_id[threadId + 16];
			}
		}  
		__syncthreads();
	  
		if(threadId < 8 && threadId + 8 < 32){
			if(threads_max[threadId] < threads_max[threadId + 8]){
				threads_max[threadId] = threads_max[threadId + 8];
				threads_id[threadId] = threads_id[threadId + 8];
			}
		}  
		__syncthreads();
	  
		if(threadId < 4 && threadId + 4 < 32){
			if(threads_max[threadId] < threads_max[threadId + 4]){
				threads_max[threadId] = threads_max[threadId + 4];
				threads_id[threadId] = threads_id[threadId + 4];
			}
		}  
		__syncthreads();
	  
		if(threadId < 2 && threadId + 2 < 32){
			if(threads_max[threadId] < threads_max[threadId + 2]){
				threads_max[threadId] = threads_max[threadId + 2];
				threads_id[threadId] = threads_id[threadId + 2];
			}
		}  
		__syncthreads();
	
		if(threadId < 1 && threadId + 1 < 32){
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


__device__ void atomicMin(float* const address, const float value)
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


__global__ void findMin(const float* __restrict__ input, const int size, float * out, int * outIdx)
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


void nelderMead_findBest(int dimension, int numberBlocks,float &best, int &index_best, float * p_obj_function, float * obj, int * idx, thrust::device_vector<float> &d_obj_function){

	findMin <<< numberBlocks, 32 >>>(p_obj_function, dimension + 1, obj, idx);
	cudaDeviceSynchronize();
	
	best = *obj;
	index_best = *idx;
}

void nelderMead_findWorst(int dimension, int numberBlocks, float &worst, int &index_worst, float * p_obj_function, float * obj, int * idx, thrust::device_vector<float> &d_obj_function){

	findMax <<< numberBlocks, 32 >>>(p_obj_function, dimension + 1, obj, idx);
	cudaDeviceSynchronize();
	
	worst = *obj;
	index_worst = *idx;
	
}


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


__global__ void nelderMead_centroid(int dimension, int index_worst, float * p_simplex, float * p_centroid){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadsMax = dimension + 1;

	int stride = threadId * dimension;	
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
		p_centroid[blockId] = (threads_sum[0] - p_simplex[index_worst * dimension + blockId]) / (dimension);
	}
}

__global__ void nelderMead_reflection(int dimension, float reflection_coef, float * p_simplex, int index_worst, float * p_centroid, float * p_reflection){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 


	if(index < dimension){
		p_reflection[index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ index_worst * dimension + index]);
	}
}

__global__ void nelderMead_expansion(int dimension, float expansion_coef, float * p_centroid, float * p_reflection, float * p_expansion){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	if(index < dimension){
		p_expansion[index] = p_centroid[index] + expansion_coef * (p_reflection[index] - p_centroid[index]);
	}
}

__global__ void nelderMead_replacement(int dimension, float * p_simplex, float * p_new_vertex, int index_worst, float * p_obj_function, float obj){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 
	int stride = index_worst * dimension;


	if(index < dimension){
		p_simplex[stride + index] = p_new_vertex[index];
	}

	if(blockId == 0 and threadId == 0){
		p_obj_function[index_worst] = obj;
	}
}

__global__ void nelderMead_contraction(int dimension, float contraction_coef, float * p_centroid, int index, float * p_simplex, float * p_vertex){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int i = blockId * 32 + threadId; 

	if(i < dimension){
		p_vertex[i] = p_centroid[i] + contraction_coef * (p_simplex[index * dimension + i] - p_centroid[i]);
	}
}

__global__ void nelderMead_shrink(int dimension, float shrink_coef, float * p_simplex, int index_best){

    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	
	int stride_best = index_best * dimension;

    int stride = blockId * dimension;

	if(threadId != index_best){
		p_simplex[stride +  threadId] = shrink_coef * p_simplex[stride_best + threadId] + (1.0f - shrink_coef) * p_simplex[stride + threadId];
	}

}


struct count_better_than_reflection{
    float reflection;

    count_better_than_reflection(float _reflection){
        reflection = _reflection;
	}
    
	__device__
	bool operator()(const float &x)
	{
		return reflection < x;
	}
};


void nelderMead_calculate_from_host(int blocks, NelderMead &p, void * h_problem_p, float * p_simplex, float * p_obj_function){

	p.evaluations_used += blocks;

	if(p.problem_type == AB_OFF_LATTICE){
		
		ABOffLattice * h_problem_parameters = (ABOffLattice*)h_problem_p;
		int threads = (*h_problem_parameters).protein_length - 2;

		calculateABOffLattice<<< blocks, threads >>>(p.dimension, h_problem_parameters->protein_length, p_simplex, p_obj_function);
		cudaDeviceSynchronize();
		
	}else if(p.problem_type == BENCHMARK){


		int threads = p.dimension;
		
		switch(p.benchmark_problem){
			case SQUARE:
				calculateSquare<<< blocks, threads >>>(p.dimension, p_simplex, p_obj_function);
				cudaDeviceSynchronize();
				break;
			case SUM:
				calculateAbsoluteSum<<< blocks, threads >>>(p.dimension, p_simplex, p_obj_function);
				cudaDeviceSynchronize();
				break;
		}
	}
}




NelderMeadResult nelderMead (NelderMead &parameters, void * problem_parameters = NULL)
{

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 2.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;
	
	thrust::device_vector<float> d_start(dimension);
	
	thrust::device_vector<float> d_simplex(dimension * (dimension + 1));
	
	thrust::device_vector<float> d_centroid(dimension);
	thrust::device_vector<float> d_reflection(dimension);
	thrust::device_vector<float> d_vertex(dimension);
	
	thrust::device_vector<float> d_obj_function(dimension + 1);
	thrust::host_vector<float>	 h_obj_function(dimension + 1);
	
	float best, worst, obj_reflection, obj_vertex;
	int index_best, index_worst;

	float * p_start 		= thrust::raw_pointer_cast(&d_start[0]);
	float * p_simplex 		= thrust::raw_pointer_cast(&d_simplex[0]);
	float * p_centroid 		= thrust::raw_pointer_cast(&d_centroid[0]);
	float * p_reflection 	= thrust::raw_pointer_cast(&d_reflection[0]);
	float * p_vertex 		= thrust::raw_pointer_cast(&d_vertex[0]);
	
	float * p_obj_function 	= thrust::raw_pointer_cast(&d_obj_function[0]);

	thrust::host_vector<float> h_vertex(dimension);

	float * obj;
	cudaMallocManaged(&obj, sizeof(float));
	cudaMemset(obj, 0.0f, sizeof(float));
	
	int * idx;
	cudaMallocManaged(&idx, sizeof(int));
	cudaMemset(idx, 0, sizeof(int));

	int * count;
	cudaMallocManaged(&count, sizeof(int));
	cudaMemset(count, 0, sizeof(int));


	thrust::copy(parameters.p_start, parameters.p_start + dimension, d_start.begin());	
	
	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();

	nelderMead_calculate_from_host(dimension + 1, parameters, problem_parameters, p_simplex, p_obj_function);
	
	int numberBlocks = ceil((float)dimension / 32.0f);

	*idx = index_best = index_worst = 0;
	*obj = best = worst = d_obj_function[0];

	nelderMead_findBest(dimension, numberBlocks, best, index_best, p_obj_function, obj, idx, d_obj_function);
	
	for (int k = 0; k < parameters.iterations_number; k++) {

		*obj = best;
		nelderMead_findWorst(dimension, numberBlocks, worst, index_worst, p_obj_function, obj, idx, d_obj_function);

		nelderMead_centroid<<< dimension, dimension + 1>>>(dimension, index_worst, p_simplex, p_centroid);
		cudaDeviceSynchronize();
		
		nelderMead_reflection<<< numberBlocks, 32 >>>(dimension, parameters.reflection_coef, p_simplex, index_worst, p_centroid, p_reflection);
		cudaDeviceSynchronize();
		
		
		nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_reflection, p_reflection, problem_parameters, obj);

		if(obj_reflection < best){
			
			nelderMead_expansion<<< numberBlocks, 32 >>>(dimension, parameters.expansion_coef, p_centroid, p_reflection, p_vertex);
			cudaDeviceSynchronize();
			
			nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters, obj);

			if(obj_vertex < best){

				nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_vertex, index_worst, p_obj_function, obj_vertex);
				cudaDeviceSynchronize();
			}else{
				nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection);
				cudaDeviceSynchronize();
			}
		}else{
			*count = 0;
			countIf<<< numberBlocks, 32 >>>(count, p_obj_function, dimension, obj_reflection);
			cudaDeviceSynchronize();

			/* Se reflection melhor que segundo pior vértice (e pior) */
			if(*count >= 2){

				nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection);
				cudaDeviceSynchronize();
			}else{
				if(obj_reflection < worst){
					
					nelderMead_contraction<<< numberBlocks, 32 >>>(dimension, parameters.contraction_coef, p_centroid, 0, p_reflection, p_vertex);
					cudaDeviceSynchronize();
				}else{
					nelderMead_contraction<<< numberBlocks, 32 >>>(dimension, parameters.contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);
					cudaDeviceSynchronize();
				}
				nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters, obj);

				if(obj_vertex < obj_reflection and obj_vertex < worst){
					
					nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_vertex, index_worst, p_obj_function, obj_vertex);
					cudaDeviceSynchronize();

				}else if(obj_reflection < worst){
					
					nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, index_worst, p_obj_function, obj_reflection);
					cudaDeviceSynchronize();
				}else{
					nelderMead_shrink<<< dimension + 1, dimension >>>(dimension, parameters.shrink_coef, p_simplex, index_best);
					cudaDeviceSynchronize();
					
					nelderMead_calculate_from_host(dimension + 1, parameters, problem_parameters, p_simplex, p_obj_function);

					*obj = best;
					*idx = index_best;
					nelderMead_findBest(dimension, numberBlocks, best, index_best, p_obj_function, obj, idx, d_obj_function);
				}
			}
		}
		if (d_obj_function[index_worst] < best){ 
			best = d_obj_function[index_worst]; 
			index_best = index_worst; 

		}
	}
	
	NelderMeadResult result;

	result.best = best;
	result.best_vertex.resize(dimension);
	result.evaluations_used = parameters.evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = d_simplex[index_best * dimension + i];
	}

	return result;
}

#endif

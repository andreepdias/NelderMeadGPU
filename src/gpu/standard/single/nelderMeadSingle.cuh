#pragma once

#include "../../shared/util.cuh"
#include "../nelderMeadShared.cuh"
#include "../../shared/print.cuh"
#include "../../shared/objectiveFunctions.cuh"

__global__ void nelderMead_reflectionSingle(int dimension, float reflection_coef, float * p_simplex, uint * p_indexes, float * p_centroid, float * p_reflection){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 


	if(index < dimension){
		p_reflection[index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ p_indexes[dimension] * dimension + index]);
	}
}

__global__ void nelderMead_expansionSingle(int dimension, float expansion_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	if(index < dimension){
		p_expansion[index] = p_reflection[index] + expansion_coef * (p_reflection[index] - p_centroid[index]);
	}
}

__global__ void nelderMead_contractionSingle(int dimension, float contraction_coef, float * p_centroid, float * p_vertex, int stride, float * p_contraction){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	if(index < dimension){
		p_contraction[index] = p_centroid[index] + contraction_coef * (p_vertex[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_replacementSingle(int dimension, float * p_simplex, float * p_new_vertex, uint * p_indexes, float * p_objective_function, float * p_obj){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 
	int stride = p_indexes[dimension] * dimension;


	if(index < dimension){
		p_simplex[stride + index] = p_new_vertex[index];
	}

	if(blockId == 0 and threadId == 0){
		p_objective_function[dimension] = p_obj[0];
	}
}

__global__ void nelderMead_updateSingle(int k, int dimension, int numberBlocks, int * p_evaluations, float expansion_coef, float contraction_coef, float shrink_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion, float * p_contraction, uint * p_indexes, float * p_objective_function, float * p_obj_reflection, float * p_obj_expansion, float * p_obj_contraction, void * d_problem_parameters, ProblemEnum problem_type, BenchmarkProblemEnum benchmark_problem, int * p_count){

	float best = p_objective_function[0];
	float worst = p_objective_function[dimension];
	float reflection = p_obj_reflection[0];
	float c;

	if(reflection < best){
		// /*c*/ p_count[0] += 1;
		
		nelderMead_expansionSingle<<< numberBlocks, 32 >>>(dimension, expansion_coef, p_simplex, p_centroid, p_reflection, p_expansion);
		cudaDeviceSynchronize();
		
		// nelderMead_calculateSingleFromDevice(d_problem_parameters, p_expansion, p_obj_expansion);
		nelderMead_calculateFromDevice(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_expansion, p_obj_expansion);

		/*e*/ p_evaluations[0] += 1;
		
		if(p_obj_expansion[0] < best){
			nelderMead_replacementSingle<<< numberBlocks, 32 >>>(dimension, p_simplex, p_expansion, p_indexes, p_objective_function, p_obj_expansion);
			cudaDeviceSynchronize();
		}else{
			nelderMead_replacementSingle<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
			cudaDeviceSynchronize();
		}
		
	}else if(reflection < p_objective_function[dimension - 1]){
		// /*c*/ p_count[1] += 1;
		
		nelderMead_replacementSingle<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
		cudaDeviceSynchronize();
	}else{
		if(reflection < worst){
			nelderMead_contractionSingle<<< numberBlocks, 32 >>>(dimension, contraction_coef, p_centroid, p_reflection, 0, p_contraction);
			cudaDeviceSynchronize();
		}else{
			nelderMead_contractionSingle<<< numberBlocks, 32 >>>(dimension, contraction_coef, p_centroid, p_simplex, p_indexes[dimension] * dimension, p_contraction);
			cudaDeviceSynchronize();
		}
		// nelderMead_calculateSingleFromDevice(d_problem_parameters, p_contraction, p_obj_contraction);
		nelderMead_calculateFromDevice(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_contraction, p_obj_contraction);
		/*e*/ p_evaluations[0] += 1;


		c = p_obj_contraction[0];

		if(c < reflection and c < worst){
			// /*c*/ p_count[2] += 1;
			
			nelderMead_replacementSingle<<< numberBlocks, 32 >>>(dimension, p_simplex, p_contraction, p_indexes, p_objective_function, p_obj_contraction);
			cudaDeviceSynchronize();
		}else if(reflection < worst){

			nelderMead_replacementSingle<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
			cudaDeviceSynchronize();
		}else{
			// /*c*/ p_count[3] += 1;
			
			nelderMead_shrink<<< dimension, dimension >>>(dimension, shrink_coef, p_simplex, p_indexes);
			cudaDeviceSynchronize();

			sequence(p_indexes, dimension + 1);
			nelderMead_calculateFromDevice(dimension + 1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_simplex, p_objective_function);
			/*e*/ p_evaluations[0] += dimension + 1;
		}
	}
}


NelderMeadResult nelderMeadSingle(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 1.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;
	
	thrust::device_vector<float> d_start(dimension);
	thrust::device_vector<float> d_simplex(dimension * (dimension + 1));	
	thrust::device_vector<float> d_centroid(dimension);
	thrust::device_vector<float> d_reflection(dimension);
	thrust::device_vector<float> d_expansion(dimension);
	thrust::device_vector<float> d_contraction(dimension);
	
    thrust::device_vector<float> d_obj_function(dimension + 1);
	thrust::device_vector<uint>  d_indexes(dimension + 1);

	thrust::device_vector<float> d_obj_reflection(1);	
	thrust::device_vector<float> d_obj_expansion(1);	
	thrust::device_vector<float> d_obj_contraction(1);
	
	thrust::device_vector<int> d_evaluations(1);
	/*c*/thrust::device_vector<int> d_count(4);

	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
	float * p_simplex			 		   = thrust::raw_pointer_cast(&d_simplex[0]);
	float * p_centroid 			 		   = thrust::raw_pointer_cast(&d_centroid[0]);
	float * p_reflection 		 		   = thrust::raw_pointer_cast(&d_reflection[0]);
	float * p_expansion 		 		   = thrust::raw_pointer_cast(&d_expansion[0]);
	float * p_contraction		 		   = thrust::raw_pointer_cast(&d_contraction[0]);
	
	float * p_objective_function 		   = thrust::raw_pointer_cast(&d_obj_function[0]);
	uint  * p_indexes 				       = thrust::raw_pointer_cast(&d_indexes[0]);
	
	float * p_obj_reflection 			   = thrust::raw_pointer_cast(&d_obj_reflection[0]);
	float * p_obj_expansion 			   = thrust::raw_pointer_cast(&d_obj_expansion[0]);
	float * p_obj_contraction 			   = thrust::raw_pointer_cast(&d_obj_contraction[0]);
	
	int * p_evaluations 				   = thrust::raw_pointer_cast(&d_evaluations[0]);
	/*c*/ int * p_count 						   = thrust::raw_pointer_cast(&d_count[0]);

	/*e*/ int evaluations_used = 0;
	
	thrust::copy(parameters.p_start, parameters.p_start + dimension, d_start.begin());
	thrust::sequence(d_indexes.begin(), d_indexes.end());

	/*c*/ thrust::fill(d_count.begin(), d_count.end(), 0);

	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();

	nelderMead_calculateFromHost(dimension + 1, parameters, h_problem_parameters, p_simplex, p_objective_function);
	/*e*/ evaluations_used += dimension + 1;

	thrust::sort_by_key(d_obj_function.begin(), d_obj_function.end(), d_indexes.begin());
	
	int numberBlocks = ceil(dimension / 32.0f);
	
	for(int i = 0; i < parameters.iterations_number; i++){
		
		nelderMead_centroid<<< dimension, dimension >>>(dimension, p_simplex, p_indexes, p_centroid);
		cudaDeviceSynchronize();
		
		nelderMead_reflectionSingle<<< numberBlocks, 32 >>>(dimension, parameters.reflection_coef, p_simplex, p_indexes, p_centroid, p_reflection);
		cudaDeviceSynchronize();
		
		// d_obj_reflection[0] = 0;
		// nelderMead_calculateSingleFromHost(h_problem_parameters, p_reflection, p_obj_reflection);
		nelderMead_calculateFromHost(1, parameters, h_problem_parameters, p_reflection, p_obj_reflection);
		/*e*/ evaluations_used += 1;
		
		nelderMead_updateSingle<<< 1, 1 >>>(i, dimension, numberBlocks, p_evaluations, parameters.expansion_coef, parameters.contraction_coef, parameters.shrink_coef, p_simplex, p_centroid, p_reflection, p_expansion, p_contraction, p_indexes, p_objective_function, p_obj_reflection, p_obj_expansion, p_obj_contraction, d_problem_parameters, parameters.problem_type, parameters.benchmark_problem, p_count);
		cudaDeviceSynchronize();
		
		thrust::sort_by_key(d_obj_function.begin(), d_obj_function.end(), d_indexes.begin());
	}

	/*e*/ evaluations_used += thrust::reduce(d_evaluations.begin(), d_evaluations.end(), 0, thrust::plus<int>());


	NelderMeadResult result;

	result.best = d_obj_function[0];
	result.best_vertex.resize(dimension);
	result.evaluations_used = evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = d_simplex[d_indexes[0] * dimension + i];
	}

	// /*c*/ thrust::host_vector<int> h_count = d_count;
	// /*c*/ printf("case 1: %d, case 2: %d, case 3: %d, case 4: %d\n", h_count[0], h_count[1], h_count[2], h_count[3]);
	
	return result;
}

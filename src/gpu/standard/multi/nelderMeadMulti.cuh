#pragma once

#include "../../shared/util.cuh"
#include "../nelderMeadShared.cuh"


__global__ void nelderMead_reflectionMulti(int p, int dimension, float reflection_coef, float * p_simplex, uint * p_indexes, float * p_centroid, float * p_reflection){

	int index = threadIdx.x;
	int blockId = blockIdx.x;
	int stride = blockId * dimension;

	if(index < dimension){
		//dimension - p + 1 + blockId
		p_reflection[stride + index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ p_indexes[dimension - blockId] * dimension + index]);
	}
}

__global__ void nelderMead_expansionMulti(int processor, int dimension, float expansion_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion){

	int index = threadIdx.x;
	int stride = processor * dimension;

	if(index < dimension){
		p_expansion[stride + index] = p_reflection[stride + index] + expansion_coef * (p_reflection[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_contractionMulti(int processor, int dimension, float contraction_coef, float * p_centroid, float * p_vertex, int stride, float * p_contraction){

	int index = threadIdx.x;
	int stride_contraction = processor * dimension;

	if(index < dimension){
		p_contraction[stride_contraction + index] = p_centroid[index] + contraction_coef * (p_vertex[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_replacementMulti(int processor, int dimension, float * p_simplex, float * p_new_vertex, uint * p_indexes, float * p_objective_function, float * p_obj){

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

__global__ void nelderMead_updateMulti(int p, int dimension, int * p_evaluations, float expansion_coef, float contraction_coef, float shrink_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion, float * p_contraction, uint * p_indexes, float * p_objective_function, float * p_obj_reflection, float * p_obj_expansion, float * p_obj_contraction, bool * p_need_shrink, void * d_problem_parameters, ProblemEnum problem_type, BenchmarkProblemEnum benchmark_problem, int * p_count1, int * p_count2, int * p_count3, int * p_count4){

	int blockId = blockIdx.x;
	float worst = p_objective_function[dimension - blockId];
	float next_worst = p_objective_function[dimension - blockId - 1];
	float best = p_objective_function[0];
	float reflection = p_obj_reflection[blockId];

	// cooperative_groups::grid_group g = cooperative_groups::this_grid();
	// g.sync();

	p_need_shrink[blockId] = false;

	if(reflection < best){
		// /*c*/ p_count1[blockId] += 1;

		nelderMead_expansionMulti<<< 1, dimension >>>(blockId, dimension, expansion_coef, p_simplex, p_centroid, p_reflection, p_expansion);
        cudaDeviceSynchronize();
		
		nelderMead_calculateFromDevice(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_expansion, p_obj_expansion, true, blockId);
		cudaDeviceSynchronize();
		// /*e*/ p_evaluations[blockId] += 1;
		
		if(p_obj_expansion[blockId] < p_objective_function[0]){
			nelderMead_replacementMulti<<< 1, dimension >>>(blockId, dimension, p_simplex, p_expansion, p_indexes, p_objective_function, p_obj_expansion);
            cudaDeviceSynchronize();
		}else{
            nelderMead_replacementMulti<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
            cudaDeviceSynchronize();
		}
	}
	else if(reflection < next_worst){
		// /*c*/ p_count2[blockId] += 1;

        nelderMead_replacementMulti<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
        /*s*/ cudaDeviceSynchronize();
	}else{

		bool is_reflection_better = false;

		if(reflection < worst){
			is_reflection_better = true;

			nelderMead_contractionMulti<<< 1, dimension >>>(blockId, dimension, contraction_coef, p_centroid, p_reflection, blockId * dimension, p_contraction);
			/*s*/ cudaDeviceSynchronize();
		}else{
			nelderMead_contractionMulti<<< 1, dimension >>>(blockId, dimension, contraction_coef, p_centroid, p_simplex, p_indexes[dimension - blockId] * dimension, p_contraction);
			/*s*/ cudaDeviceSynchronize();
		}

		cudaDeviceSynchronize();
		nelderMead_calculateFromDevice(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_contraction, p_obj_contraction, true, blockId);
		// /*e*/ p_evaluations[blockId] += 1;

		float contraction = p_obj_contraction[blockId];

		if(contraction < worst and contraction < reflection){
			// /*c*/ p_count3[blockId] += 1;

			nelderMead_replacementMulti<<< 1, dimension >>>(blockId, dimension, p_simplex, p_contraction, p_indexes, p_objective_function, p_obj_contraction);
            /*s*/ cudaDeviceSynchronize();
		}else{
			// /*c*/ p_count4[blockId] += 1;

			p_need_shrink[blockId] = true;

			if(is_reflection_better){
				nelderMead_replacementMulti<<< 1, dimension >>>(blockId, dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
				/*s*/ cudaDeviceSynchronize();
			}

		}
	}
}

NelderMeadResult nelderMeadMulti(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

	int p = parameters.p;

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 1.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;
	
	thrust::device_vector<float> d_start(dimension);	
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

	thrust::device_vector<int> d_evaluations(p);
	thrust::device_vector<int> d_count1(p);
	thrust::device_vector<int> d_count2(p);
	thrust::device_vector<int> d_count3(p);
	thrust::device_vector<int> d_count4(p);

	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
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

	int * p_evaluations 				   = thrust::raw_pointer_cast(&d_evaluations[0]);
	int * p_count1 						   = thrust::raw_pointer_cast(&d_count1[0]);
	int * p_count2 						   = thrust::raw_pointer_cast(&d_count2[0]);
	int * p_count3 						   = thrust::raw_pointer_cast(&d_count3[0]);
	int * p_count4 						   = thrust::raw_pointer_cast(&d_count4[0]);

	int evaluations_used = 0;
	
	thrust::copy(parameters.p_start, parameters.p_start + dimension, d_start.begin());	
	thrust::sequence(d_indexes.begin(), d_indexes.end());

	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();

	nelderMead_calculateFromHost(dimension + 1, parameters, h_problem_parameters, p_simplex, p_objective_function);
	// /*e*/ evaluations_used += dimension + 1;
	
	thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	
	for(int i = 0; i < parameters.iterations_number; i++){
		nelderMead_centroid<<< dimension, dimension + 1 - p >>>(dimension, p_simplex, p_indexes, p_centroid, p);
		cudaDeviceSynchronize();
		
		nelderMead_reflectionMulti<<< p, dimension >>>(p, dimension, parameters.reflection_coef, p_simplex, p_indexes, p_centroid, p_reflection);
		cudaDeviceSynchronize();
		
		nelderMead_calculateFromHost(p, parameters, h_problem_parameters, p_reflection, p_obj_reflection);
		// /*e*/ evaluations_used += p;
		
		nelderMead_updateMulti<<< p, 1 >>>(p, dimension, p_evaluations, parameters.expansion_coef, parameters.contraction_coef, parameters.shrink_coef, p_simplex, p_centroid, p_reflection, p_expansion, p_contraction, p_indexes, p_objective_function, p_obj_reflection, p_obj_expansion, p_obj_contraction, p_need_shrink, d_problem_parameters, parameters.problem_type, parameters.benchmark_problem, p_count1, p_count2, p_count3, p_count4);
		cudaDeviceSynchronize();
		
		bool need_shrink = thrust::any_of(d_need_shrink.begin(), d_need_shrink.end(), thrust::identity<bool>());
		
		if(need_shrink){
			nelderMead_shrink<<< dimension, dimension >>>(dimension, parameters.shrink_coef, p_simplex, p_indexes);
			cudaDeviceSynchronize();
			
			thrust::sequence(d_indexes.begin(), d_indexes.end());
			nelderMead_calculateFromHost(dimension + 1, parameters, h_problem_parameters, p_simplex, p_objective_function);
			// /*e*/ evaluations_used += dimension + 1;
		}
		
		thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	}

	// /*e*/ evaluations_used += thrust::reduce(d_evaluations.begin(), d_evaluations.end(), 0, thrust::plus<int>());

	NelderMeadResult result;

	result.best = d_objective_function[0];
	result.best_vertex.resize(dimension);
	result.evaluations_used = evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = d_simplex[d_indexes[0] * dimension + i];
	}

	// /*c*/ thrust::host_vector<int> h_count(4);
	// /*c*/ h_count[0] = thrust::reduce(d_count1.begin(), d_count1.end(), 0, thrust::plus<int>());
	// /*c*/ h_count[1] = thrust::reduce(d_count2.begin(), d_count2.end(), 0, thrust::plus<int>());
	// /*c*/ h_count[2] = thrust::reduce(d_count3.begin(), d_count3.end(), 0, thrust::plus<int>());
	// /*c*/ h_count[3] = thrust::reduce(d_count4.begin(), d_count4.end(), 0, thrust::plus<int>());
	// /*c*/ printf("case 1: %d, case 2: %d, case 3: %d, case 4: %d\n", h_count[0], h_count[1], h_count[2], h_count[3]);
	
	return result;
}

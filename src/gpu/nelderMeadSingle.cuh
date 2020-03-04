#ifndef NELMINSINGLE_H
#define NELMINSINGLE_H

#include "util.cuh"
#include "nelderMeadShared.cuh"

__global__ void nelderMead_reflection(int dimension, float reflection_coef, float * p_simplex, uint * p_indexes, float * p_centroid, float * p_reflection){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 


	if(index < dimension){
		p_reflection[index] = p_centroid[index] + reflection_coef * (p_centroid[index] - p_simplex[ p_indexes[dimension] * dimension + index]);
	}
}

__global__ void nelderMead_expansion(int dimension, float expansion_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	if(index < dimension){
		p_expansion[index] = p_reflection[index] + expansion_coef * (p_reflection[index] - p_centroid[index]);
	}
}

__global__ void nelderMead_contraction(int dimension, float contraction_coef, float * p_centroid, float * p_vertex, int stride, float * p_contraction){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int index = blockId * 32 + threadId; 

	if(index < dimension){
		p_contraction[index] = p_centroid[index] + contraction_coef * (p_vertex[stride + index] - p_centroid[index]);
	}
}

__global__ void nelderMead_replacement(int dimension, float * p_simplex, float * p_new_vertex, uint * p_indexes, float * p_objective_function, float * p_obj){

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

__global__ void nelderMead_update(int dimension, int * p_evaluations, float expansion_coef, float contraction_coef, float shrink_coef, float * p_simplex, float * p_centroid, float * p_reflection, float * p_expansion, float * p_contraction, uint * p_indexes, float * p_objective_function, float * p_obj_reflection, float * p_obj_expansion, float * p_obj_contraction, void * d_problem_parameters, ProblemEnum problem_type, BenchmarkProblemEnum benchmark_problem){

	int numberBlocks = ceil(dimension / 32.0f);

	if(p_obj_reflection[0] < p_objective_function[0]){

		nelderMead_expansion<<< numberBlocks, 32 >>>(dimension, expansion_coef, p_simplex, p_centroid, p_reflection, p_expansion);
		cudaDeviceSynchronize();
		/*p*/printVertexDevice(dimension, p_expansion, "Expansion");
		
		nelderMead_calculate_from_device(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_expansion, p_obj_expansion);
		cudaDeviceSynchronize();
		p_evaluations[0] += 1;
		printEvaluationsDevice(p_evaluations, 1);
		/*p*/printSingleObjFunctionDevice(p_obj_expansion, "Objective Function Expansion");
		
		if(p_obj_expansion[0] < p_objective_function[0]){
			nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_expansion, p_indexes, p_objective_function, p_obj_expansion);
			cudaDeviceSynchronize();
			/*p*/printSimplexDevice(dimension, p_simplex, "Replacement, Case 1a (expansion better than best vertex)");
		}else{
			nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
			cudaDeviceSynchronize();
			/*p*/printSimplexDevice(dimension, p_simplex, "Replacement, Case 1b (reflection better than best vertex)");
		}
		
	}else if(p_obj_reflection[0] < p_objective_function[dimension - 1]){
		nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_reflection, p_indexes, p_objective_function, p_obj_reflection);
		cudaDeviceSynchronize();
		/*p*/printSimplexDevice(dimension, p_simplex, "Replacement, Case 2 (reflection better than second worst vertex)");
	}else{
		if(p_obj_reflection[0] < p_objective_function[dimension]){
			/*p*/printf("---First case contraction---\n");
			nelderMead_contraction<<< numberBlocks, 32 >>>(dimension, contraction_coef, p_centroid, p_reflection, 0, p_contraction);
			cudaDeviceSynchronize();
		}else{
			/*p*/printf("---Second case contraction---\n");
			nelderMead_contraction<<< numberBlocks, 32 >>>(dimension, contraction_coef, p_centroid, p_simplex, p_indexes[dimension] * dimension, p_contraction);
			cudaDeviceSynchronize();
		}
		/*p*/printVertexDevice(dimension, p_contraction, "Contraction");
		
		nelderMead_calculate_from_device(1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_contraction, p_obj_contraction);
		cudaDeviceSynchronize();
		p_evaluations[0] += 1;
		printEvaluationsDevice(p_evaluations, 1);
		/*p*/printSingleObjFunctionDevice(p_obj_contraction, "Objective Function Contraction");
		
		
		if(p_obj_contraction[0] < p_objective_function[dimension]){
			nelderMead_replacement<<< numberBlocks, 32 >>>(dimension, p_simplex, p_contraction, p_indexes, p_objective_function, p_obj_contraction);
			cudaDeviceSynchronize();
			/*p*/printSimplexDevice(dimension, p_simplex, "Replacement, Case 3a (contraction better than worst vertex)");
		}else{
			/*p*/printSimplexDevice(dimension, p_simplex, "Pre Shrink");
			nelderMead_shrink<<< dimension, dimension >>>(dimension, shrink_coef, p_simplex, p_indexes);
			cudaDeviceSynchronize();
			/*p*/printSimplexDevice(dimension, p_simplex, "Shrink Case 3b (contraction worst than worst vertex)");
			sequence(p_indexes, dimension + 1);
			nelderMead_calculate_from_device(dimension + 1, dimension, problem_type, benchmark_problem, d_problem_parameters, p_simplex, p_objective_function);
			cudaDeviceSynchronize();
			p_evaluations[0] += dimension + 1;
			printEvaluationsDevice(p_evaluations, dimension + 1);

		}
	}
}


NelderMeadResult nelderMead(NelderMead &parameters, void * h_problem_parameters = NULL, void * d_problem_parameters = NULL){

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 1.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;
	
	thrust::device_vector<float> d_start(dimension);
	
	//thrust::device_vector<float> d_aminoacid_position(protein_length * 3);
	//thrust::host_vector<float> 	 h_aminoacid_position(protein_length * 3);

	thrust::device_vector<float> d_simplex(dimension * (dimension + 1));	

	thrust::device_vector<float> d_centroid(dimension);
	thrust::device_vector<float> d_reflection(dimension);
	thrust::device_vector<float> d_expansion(dimension);
	thrust::device_vector<float> d_contraction(dimension);
	
    thrust::device_vector<float> d_objective_function(dimension + 1);
	thrust::device_vector<uint>  d_indexes(dimension + 1);

	thrust::device_vector<float> d_obj_reflection(1);	
	thrust::device_vector<float> d_obj_expansion(1);	
	thrust::device_vector<float> d_obj_contraction(1);

	thrust::device_vector<int> d_evaluations(1);

	float * p_start 			 		   = thrust::raw_pointer_cast(&d_start[0]);
	
	//float * p_aminoacid_position 		   = thrust::raw_pointer_cast(&d_aminoacid_position[0]);
	
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

	int * p_evaluations 				   = thrust::raw_pointer_cast(&d_evaluations[0]);

	int evaluations_used = 0;
	
	thrust::copy(parameters.p_start, parameters.p_start + dimension, d_start.begin());

	/*p*/printVertexHost(dimension, d_start, "Start");
	

	/* int, float, pointers */
	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();
	/*p*/printSimplexHost(dimension, d_simplex, "Initialize");

	thrust::sequence(d_indexes.begin(), d_indexes.end());

	nelderMead_calculate_from_host(dimension + 1, parameters, h_problem_parameters, p_simplex, p_objective_function);
	evaluations_used += dimension + 1;
	printEvaluationsHost(evaluations_used, dimension + 1);
	/*p*/printObjFunctionHost(dimension, d_objective_function, d_indexes, "Objective Function");
	
	thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
	/*p*/printObjFunctionHost(dimension, d_objective_function, d_indexes, "Objective Function Sorted");
	
	for(int i = 0; i < parameters.iterations_number; i++){
		nelderMead_centroid<<< dimension, dimension >>>(dimension, p_simplex, p_indexes, p_centroid);
		cudaDeviceSynchronize();
		/*p*/printVertexHost(dimension, d_centroid, "Centroid");
		
		int numberBlocksReflection = ceil(dimension / 32.0f);
		
		nelderMead_reflection<<< numberBlocksReflection, 32 >>>(dimension, parameters.reflection_coef, p_simplex, p_indexes, p_centroid, p_reflection);
		cudaDeviceSynchronize();
		/*p*/printVertexHost(dimension, d_reflection, "Reflection");
		
		nelderMead_calculate_from_host(1, parameters, h_problem_parameters, p_reflection, p_obj_reflection);
		evaluations_used += 1;
		printEvaluationsHost(evaluations_used, 1);
		/*p*/printSingleObjFunctionHost(dimension, d_obj_reflection, "Objective Function Reflection");
		
		nelderMead_update<<< 1, 1 >>>(dimension, p_evaluations, parameters.expansion_coef, parameters.contraction_coef, parameters.shrink_coef, p_simplex, p_centroid, p_reflection, p_expansion, p_contraction, p_indexes, p_objective_function, p_obj_reflection, p_obj_expansion, p_obj_contraction, d_problem_parameters, parameters.problem_type, parameters.benchmark_problem);
		cudaDeviceSynchronize();
		
		evaluations_used += thrust::reduce(d_evaluations.begin(), d_evaluations.end(), 0, thrust::plus<int>());
		thrust::fill(d_evaluations.begin(), d_evaluations.end(), 0);
		
		printEvaluationsHost(evaluations_used, 0);
		
		/*p*/printObjFunctionHost(dimension, d_objective_function, d_indexes, "Objective Function");
		thrust::sort_by_key(d_objective_function.begin(), d_objective_function.end(), d_indexes.begin());
		/*p*/printObjFunctionHost(dimension, d_objective_function, d_indexes, "Objective Function Sorted");
		
		/*p*/printf("------------------ END ITERATION %d ------------------\n\n", i + 1);
	}

	NelderMeadResult result;

	result.best = d_objective_function[0];
	result.best_vertex.resize(dimension);
	result.evaluations_used = evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = d_simplex[d_indexes[0] * dimension + i];
	}
	
	return result;
}

#endif
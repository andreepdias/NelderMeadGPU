#ifndef NELMIN_H
#define NELMIN_H

#include "util.cuh"


// Nelder-Mead Minimization Algorithm ASA047
// from the Applied Statistics Algorithms available
// in STATLIB. Adapted from the C version by J. Burkhardt
// http://people.sc.fsu.edu/~jburkardt/c_src/asa047/asa047.html


struct Calculate3DAB{
    int protein_length;

    float * p_vertex;
    float * p_aminoacid_position;

    Calculate3DAB(float * _p_vertex, float * _p_aminoacid_position, int _protein_length){
        p_vertex = _p_vertex, 
		p_aminoacid_position = _p_aminoacid_position, 
		protein_length = _protein_length;
	}
    
    __device__ float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d, dx, dy, dz;

        sum += (1.0f - cosf(p_vertex[id])) / 4.0f;

        for(int i = id + 2; i < protein_length; i++){

            if(aminoacid_sequence[id] == 'A' && aminoacid_sequence[i] == 'A')
                c = 1.0;
            else if(aminoacid_sequence[id] == 'B' && aminoacid_sequence[i] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = p_aminoacid_position[id] - p_aminoacid_position[i];
            dy = p_aminoacid_position[id + protein_length] - p_aminoacid_position[i + protein_length];
            dz = p_aminoacid_position[id + protein_length * 2] - p_aminoacid_position[i + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );
            
            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
                
        }
        return sum;
    }
};


__global__ void calculateCoordinates(float * p_vertex, float * p_aminoacid_position, int protein_length){

	//p_vertex on shared?

	p_aminoacid_position[0] = 0.0f;
	p_aminoacid_position[0 + protein_length] = 0.0f;
	p_aminoacid_position[0 + protein_length * 2] = 0.0f;

	p_aminoacid_position[1] = 0.0f;
	p_aminoacid_position[1 + protein_length] = 1.0f; 
	p_aminoacid_position[1 + protein_length * 2] = 0.0f;

	p_aminoacid_position[2] = cosf(p_vertex[0]);
	p_aminoacid_position[2 + protein_length] = sinf(p_vertex[0]) + 1.0f;
	p_aminoacid_position[2 + protein_length * 2] = 0.0f;

	for(int i = 3; i < protein_length; i++){
		p_aminoacid_position[i] = p_aminoacid_position[i - 1] + cosf(p_vertex[i - 2]) * cosf(p_vertex[i + protein_length - 5]); // i - 3 + protein_length - 2
		p_aminoacid_position[i + protein_length] = p_aminoacid_position[i - 1 + protein_length] + sinf(p_vertex[i - 2]) * cosf(p_vertex[i + protein_length - 5]);
		p_aminoacid_position[i + protein_length * 2] = p_aminoacid_position[i - 1 + protein_length * 2] + sinf(p_vertex[i + protein_length - 5]);
	}
}

float calculate3DABOffLattice(int dimension, float * p_vertex, void * problem_parameters){

	ABOffLattice * parametersAB = (ABOffLattice*)problem_parameters;
	int protein_length = (*parametersAB).protein_length;
	
	calculateCoordinates<<<1, 1>>>(p_vertex, (*parametersAB).p_aminoacid_postion, protein_length);
	cudaDeviceSynchronize();

	Calculate3DAB unary_op(p_vertex, p_aminoacid_position, protein_length);
	thrust::plus<float> binary_op;

	float result = thrust::transform_reduce(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(protein_length - 2), unary_op, 0.0f, binary_op);

	return result;
}


__global__ void nelderMead_initialize(int dimension, float step, float * start, float * p_simplex){

    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int stride = blockId * dimension;

	p_simplex[stride +  threadId] = start[threadId];
	
	if(threadId == blockId){
		p_simplex[stride +  threadId] = start[threadId] + step;
	}
}

void nelderMead_calculateSimplex(int dimension, int &evaluations_used, float * p_simplex, float * p_obj_function, void * problem_parameters = NULL){

	for(int i =  0; i < dimension + 1; i++){
		p_obj_function[i] = calculate3DABOffLattice(dimension, p_simplex + (i * dimension), problem_parameters);
		// p_obj_function[i] = (*fn)(p_simplex + (i * dimension));
		evaluations_used += 1;
	}
}

void nelderMead_calculateVertex(int dimension, int &evaluations_used, float &obj, float * p_vertex, void * problem_parameters = NULL){
	
	obj = calculate3DABOffLattice(dimension, p_vertex, problem_parameters);
	evaluations_used = evaluations_used + 1;
	// obj = (*fn)( &p_vertex[0] );
}


void nelderMead_findBest(int dimension, float &best, int &index_best, thrust::device_vector<float> &d_obj_function){

	device_vector<float>::iterator it = thrust::max_element(d_obj_function.begin(), d_obj_function.end());
	
	index_best = it - d_obj_function.begin();
	best = *it;
}

void nelderMead_findWorst(int dimension, float &worst, int &index_worst, thrust::device_vector<float> &d_obj_function){

	device_vector<float>::iterator it = thrust::min_element(d_obj_function.begin(), d_obj_function.end());
	
	index_worst = it - d_obj_function.begin();
	worst = *it;
}


__global__ void nelderMead_centroid(int dimension, int index_worst, float * p_simplex, float * p_centroid){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int threadsMax = dimension;

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
		  p_centroid[blockId] = threads_sum[0] / (threadsMax);
	}
}


void nelderMead_reflection(int dimension, int index_worst, float reflection_coef, float * p_simplex, float * p_centroid, float * p_reflection){

	for (int i = 0; i < dimension; i++ ) {
		p_reflection[i] = p_centroid[i] + reflection_coef * ( p_centroid[i] - p_simplex[i + index_worst * dimension] );
	}
}

void nelderMead_expansion(int dimension, float expansion_coef, float * p_centroid, float * p_reflection, float * p_expansion){

	for (int i = 0; i < dimension; i++ ) {
		p_expansion[i] = p_centroid[i] + expansion_coef * ( p_reflection[i] - p_centroid[i] );
	}
}

void nelderMead_replacement(int dimension, int index, float * p_simplex, float * p_vertex, float obj, float * p_obj_function){

	for (int i = 0; i < dimension; i++ ) { 
		p_simplex[index * dimension + i] = p_vertex[i]; 
	}
	p_obj_function[index] = obj;
}

void nelderMead_contraction(int dimension, float contraction_coef, float * p_centroid, int index, float * p_simplex, float * p_vertex){

	for (int i = 0; i < dimension; i++ ) {
	 	p_vertex[i] = p_centroid[i] + contraction_coef * ( p_simplex[index * dimension + i] - p_centroid[i] );
	}
}

void nelderMead_shrink(int dimension, int index_best, float * p_simplex){

	int stride_best = index_best * dimension;

	for(int i = 0; i < dimension + 1; i++){

		int stride = i * dimension;
		for(int j = 0; j < dimension; j++){
			 p_simplex[stride + j] = (p_simplex[stride + j] + p_simplex[stride_best + j]) * 0.5f;
			// p_simplex[stride + j] = 0.5 * p_simplex[stride_best + j] + (1.0 - 0.5) * p_simplex[stride + j];
		}
	}
}

NelderMeadResult nelderMead (NelderMead &parameters, void * problem_parameters = NULL, void * d_problem_parameters = NULL)
{

	int dimension = parameters.dimension;

	parameters.step = 1.0f;
	parameters.reflection_coef = 1.0f;
	parameters.expansion_coef = 2.0f;
	parameters.contraction_coef = 0.5f;
	parameters.shrink_coef = 0.5f;

	parameters.evaluations_used = 0;
	
	thrust::device_vector<float> d_start(dimension)
	
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


	nelderMead_initialize<<< dimension + 1, dimension >>>(dimension, parameters.step, p_start, p_simplex);
	cudaDeviceSynchronize();

	nelderMead_calculateSimplex(dimension, parameters.evaluations_used, p_simplex, p_obj_function, problem_parameters);

	nelderMead_findBest(dimension, best, index_best, p_obj_function);

	for (int k = 0; k < parameters.iterations_number; k++) {

		nelderMead_findWorst(dimension, worst, index_worst, p_obj_function);
		
		nelderMead_centroid(dimension, index_worst, p_simplex, p_centroid);

		nelderMead_reflection(dimension, index_worst, parameters.reflection_coef, p_simplex, p_centroid, p_reflection);
		nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_reflection, p_reflection, problem_parameters);
	


		if(obj_reflection < best){
			nelderMead_expansion(dimension, parameters.expansion_coef, p_centroid, p_reflection, p_vertex);
			nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters);

			if(obj_vertex < best){
				nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
			}else{
				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}
		}else{
			int c = 0;
			for(int i = 0; i < dimension + 1; i++){
				if(obj_reflection < p_obj_function[i]){
					c++;
				}
			}

			/* Se reflection melhor que segundo pior vértice (e pior) */
			if(c >= 2){
				nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
			}else{

				if(obj_reflection < worst){
					nelderMead_contraction(dimension, parameters.contraction_coef, p_centroid, 0, p_reflection, p_vertex);
				}else{
					nelderMead_contraction(dimension, parameters.contraction_coef, p_centroid, index_worst, p_simplex, p_vertex);
				}
				nelderMead_calculateVertex(dimension, parameters.evaluations_used, obj_vertex, p_vertex, problem_parameters);

				 if(obj_vertex < obj_reflection and obj_vertex < worst){
					nelderMead_replacement(dimension, index_worst, p_simplex, p_vertex, obj_vertex, p_obj_function);
				 }else if(obj_reflection < worst){
					nelderMead_replacement(dimension, index_worst, p_simplex, p_reflection, obj_reflection, p_obj_function);
				}else{
					nelderMead_shrink(dimension, index_best, p_simplex);
					nelderMead_calculateSimplex(dimension, parameters.evaluations_used, p_simplex, p_obj_function, problem_parameters);

					nelderMead_findBest(dimension, best, index_best, p_obj_function);
				}
			}
		}

		if (p_obj_function[index_worst] < best){ 
			best = p_obj_function[index_worst]; 
			index_best = index_worst; 
		}
	}
	
	NelderMeadResult result;

	result.best = best;
	result.best_vertex.resize(dimension);
	result.evaluations_used = parameters.evaluations_used;

	for(int i = 0; i < dimension; i++){
		result.best_vertex[i] = p_simplex[index_best * dimension + i];
	}

	return result;
}

#endif

#include "util.cuh"


__device__ float3 operator-(const float3 &a, const float3 &b) {
	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

struct Calculate3DAB{
    int protein_length;

    float * p_vertex;
    float3 * p_aminoacid_position;

    Calculate3DAB(float * _p_vertex, float3 * _p_aminoacid_position, int _protein_length){
        p_vertex = _p_vertex, 
		p_aminoacid_position = _p_aminoacid_position, 
		protein_length = _protein_length;
	}
    
    __device__ float operator()(const unsigned int& id) const { 

        float sum = 0.0f, c, d;

		sum += (1.0f - cosf(p_vertex[id])) / 4.0f;
		
		float3 pos_id = p_aminoacid_position[id];
		float3 pos_i;
		float3 r;

		char aa_id, aa_i;

        for(int i = id + 2; i < protein_length; i++){

			aa_id = aminoacid_sequence[id];
			aa_i = aminoacid_sequence[i];

            if(aa_id == 'A' && aa_i == 'A')
                c = 1.0;
            else if(aa_id == 'B' && aa_i == 'B')
                c = 0.5;
            else
                c = -0.5;

			pos_i = p_aminoacid_position[i];

			r = pos_id - pos_i;

            d = sqrtf( (r.x * r.x) + (r.y * r.y) + (r.z * r.z) );
            
            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );
                
        }
        return sum;
    }
};

void calculateCoordinatesHost(thrust::host_vector<float> &h_vertex, thrust::host_vector<float> &h_aminoacid_position, int protein_length){

	h_aminoacid_position[0] = 0.0f;
	h_aminoacid_position[1] = 0.0f;
	h_aminoacid_position[2] = 0.0f;

	h_aminoacid_position[3] = 0.0f;
	h_aminoacid_position[4] = 1.0f; 
	h_aminoacid_position[5] = 0.0f;

	h_aminoacid_position[6] = cosf(h_vertex[0]);
	h_aminoacid_position[7] = sinf(h_vertex[0]) + 1.0f;
	h_aminoacid_position[8] = 0.0f;

	for(int i = 9; i < protein_length * 3; i += 3){
		h_aminoacid_position[i] = h_aminoacid_position[i - 3] + cosf(h_vertex[(i / 3) - 2]) * cosf(h_vertex[(i / 3) + protein_length - 5]); // i - 3 + protein_length - 2
		h_aminoacid_position[i + 1] = h_aminoacid_position[i - 2] + sinf(h_vertex[(i / 3) - 2]) * cosf(h_vertex[(i / 3) + protein_length - 5]);
		h_aminoacid_position[i + 2] = h_aminoacid_position[i - 1] + sinf(h_vertex[(i / 3) + protein_length - 5]);
	}
}

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


__global__ void calculateSingleAB3D(float * obj, const int protein_length, const float * vertex){

	int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	// int threadsMax = protein_length - 2;
	int index = blockId + threadId + 2;

	float sum = 0.0f, c, d;
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
	
		aminoacid_position[2].x = cosf(vertex[0]);
		aminoacid_position[2].y = sinf(vertex[0]) + 1.0f;
		aminoacid_position[2].z = 0.0f;
	
		for(int i = 3; i < protein_length; i++){
			aminoacid_position[i].x = aminoacid_position[i - 1].x + cosf(vertex[i - 2]) * cosf(vertex[i + protein_length - 5]); // i - 3 + protein_length - 2
			aminoacid_position[i].y = aminoacid_position[i - 1].y + sinf(vertex[i - 2]) * cosf(vertex[i + protein_length - 5]);
			aminoacid_position[i].z = aminoacid_position[i - 1].z + sinf(vertex[i + protein_length - 5]);
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

void calculate3DABOffLattice(int dimension, float * p_vertex, void * problem_parameters, float * obj){
	
	ABOffLattice * parametersAB = (ABOffLattice*)problem_parameters;
	int protein_length = (*parametersAB).protein_length;

	*obj = 0;
	calculateSingleAB3D<<<protein_length - 2, protein_length - 2>>>(obj, protein_length, p_vertex);
	cudaDeviceSynchronize();

}

__global__ void calculateABOffLattice(const int dimension, const int protein_length, const float * p_simplex, float * p_objective_function){

	int threadId = threadIdx.x;

    int stride = blockIdx.x * dimension;
	int threadsMax = protein_length - 2;

	float sum = 0.0f, c, d;
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
	
		aminoacid_position[2].x = cosf(p_simplex[stride + 0]);
		aminoacid_position[2].y = sinf(p_simplex[stride + 0]) + 1.0f;
		aminoacid_position[2].z = 0.0f;
	
		for(int i = 3; i < protein_length; i++){
			aminoacid_position[i].x = aminoacid_position[i - 1].x + cosf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]); // i - 3 + protein_length - 2
			aminoacid_position[i].y = aminoacid_position[i - 1].y + sinf(p_simplex[stride + i - 2]) * cosf(p_simplex[stride + i + protein_length - 5]);
			aminoacid_position[i].z = aminoacid_position[i - 1].z + sinf(p_simplex[stride + i + protein_length - 5]);
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
  
		p_objective_function[blockIdx.x] = threads_sum[0];
	}
}



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

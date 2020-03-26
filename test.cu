#include <thrust/device_vector.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>


__device__ void atomicMax(float* const address, const float value, int* const addressIdx, const int idx)
{
	if (*address >= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI;
	int assumed;

	int oldIdx = *addressIdx;
	int assumedIdx;

	do 
	{
		assumed = old;
		assumedIdx = oldIdx;

		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
		oldIdx = atomicCAS(addressIdx, assumedIdx, idx);
	} while (assumed != old and assumedIdx != oldIdx);
}

__global__ void findMax(const float* __restrict__ input, const int size, float * out, int * outIdx)
{
    __shared__ float sharedMax;
    __shared__ int sharedMaxIdx;

	int localMaxIdx = threadIdx.x + blockIdx.x * blockDim.x;
	float localMax = input[localMaxIdx];
	int threadsMax = 32;
	int threadId = threadIdx.x;

    // if (threadIdx.x == 0)
    // {
    //     sharedMax = localMax;
    //     sharedMaxIdx = localMaxIdx;
	// }
	
	__shared__ float threads_max [32];
	threads_max[threadId] = localMax;
  
	__syncthreads();

	// if(threadId < 256 && threadId + 256 < threadsMax){
	// 	threads_max[threadId] += threads_max[threadId + 256];
	// }  
	// __syncthreads();

	// if(threadId < 128 && threadId + 128 < threadsMax){
	// 	threads_max[threadId] += threads_max[threadId + 128];
	// }  
	// __syncthreads();
	// if(threadId < 64 && threadId + 64 < threadsMax){
	// 	threads_max[threadId] += threads_max[threadId + 64];
	// }  
	// __syncthreads();

	// if(threadId < 32 && threadId + 32 < threadsMax){
	// 	threads_max[threadId] += threads_max[threadId + 32];
	// }  
	// __syncthreads();
  
	if(threadId < 16 && threadId + 16 < threadsMax){
		threads_max[threadId] = max(threads_max[threadId], threads_max[threadId + 16]);
	}  
	__syncthreads();
  
	if(threadId < 8 && threadId + 8 < threadsMax){
		threads_max[threadId] = max(threads_max[threadId], threads_max[threadId + 8]);
	}  
	__syncthreads();
  
	if(threadId < 4 && threadId + 4 < threadsMax){
		threads_max[threadId] = max(threads_max[threadId], threads_max[threadId + 4]);
	}  
	__syncthreads();
  
	if(threadId < 2 && threadId + 2 < threadsMax){
		threads_max[threadId] = max(threads_max[threadId], threads_max[threadId + 2]);
	}  
	__syncthreads();

	if(threadId < 1 && threadId + 1 < threadsMax){
		threads_max[threadId] = max(threads_max[threadId], threads_max[threadId + 1]);
	}  
	__syncthreads();
	
	if(threadId == 0){
		sharedMax = (threads_max[0]);
	}
	
	//__syncthreads();

	/*
	int block = size / blockDim.x;
	int start = block * blockIdx.x;
	int end = start + block;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
			localMaxIdx = i;
        }
	}
	*/

    // atomicMax(&sharedMax, localMax, &sharedMaxIdx, localMaxIdx);

	__syncthreads();

    if (threadIdx.x == 0)
    {
		atomicMax(out, sharedMax, outIdx, sharedMaxIdx);
    }
}

double stime(){
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
    return mlsec/1000.0;
}

int main()
{
	const int N = 2560;
	
	thrust::device_vector<float> input(N);

	for(int i = 0; i < N; i ++){
		input[i] = (float)rand() / RAND_MAX;
	}

	float * out;

	int * outIdx;
	cudaMallocManaged(&out, sizeof(float));
	cudaMallocManaged(&outIdx, sizeof(int));

	*out = input[0];
	*outIdx = 0;

    std::cout << "array size=" << N << std::endl;
    std::cout << std::endl;

	cudaEvent_t start, stop;
	float elapsed_time;
	
	double fstart = stime();
	
	
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	
	findMax <<< 80, 32 >>>(thrust::raw_pointer_cast(input.data()), N, out, outIdx );
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	
	double fstop = stime();
	double felapsed_time = fstop - fstart;

	printf("out: %.7f, idx: %d\n", *out, *outIdx);
	printf("elapsed time: %.7f, %.7f\n", elapsed_time, felapsed_time);

    return 0;   
}

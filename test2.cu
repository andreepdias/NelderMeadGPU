#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;

//1
//105
//1125

// Global max reduce example based on CppCon 2016: “Bringing Clang and C++ to GPUs: An Open-Source, CUDA-Compatible GPU C++ Compiler"
__global__ void d_max_reduce(const int *in, int *out, size_t N) {
    int sum = 0;
    size_t start = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    for (size_t i = start; i < start + 4 && i < N; i++) {
        sum = max(__ldg(in + i), sum);
    }

    for (int i = 16; i; i >>= 1) {
        sum = max(__shfl_down(sum, i), sum);
    }

    __shared__ int shared_sum;
    shared_sum = 0;
    __syncthreads();

    if (threadIdx.x % 32 == 0) {
        atomicMax(&shared_sum, sum);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(out, shared_sum);
    }
}

int test_reduce(std::vector<int> &v) {
    int *in;
    int *out;

    cudaMalloc(&in, v.size() * sizeof(int));
    cudaMalloc(&out, sizeof(int));

    cudaMemcpy(in, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out, 0, sizeof(int));

    int threads = 32;

    d_max_reduce<<<1, threads / 4>>>(in, out, v.size());

    int res;

    cudaMemcpy(&res, out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);

    return res;
}
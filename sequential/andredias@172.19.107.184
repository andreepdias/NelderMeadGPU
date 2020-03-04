#include <iostream>
#include <thrust/device_vector.h>


__device__ void calculate(int * p_evaluations){
    int blockId = blockIdx.x;

    p_evaluations[blockId]++;

}

__global__ void update(int * p_evaluations){

    calculate(p_evaluations);
    calculate(p_evaluations);  

}


int main(){


    printf("1\n");
    int p = 5;
    
    printf("2\n");
    thrust::device_vector<int> d_evaluations(p);
    
    printf("3\n");
    int * p_evaluations = thrust::raw_pointer_cast(&d_evaluations[0]);
    
    printf("4\n");
    update<<< p, 1 >>> (p_evaluations);
    printf("5\n");
    cudaDeviceSynchronize();
    
    printf("6\n");
    int sum = thrust::reduce(d_evaluations.begin(), d_evaluations.end(), 0, thrust::plus<int>());
    
    printf("7\n");
    printf("%d\n", sum);
    


}
#include <iostream>


__device__ void calculate(int * p_evaluations){
    int blockId = blockIdx.x;

    p_evaluations[blockId]++;

}

__global__ void update(int * p_evaluations){

    calculate(p_evaluations);
    calculate(p_evaluations);  

}


int main(){

    int p = 5;

    thurst::device_vector<int> d_evaluations(p);

    int * p_evaluations = thrust::raw_ponter_cast(&d_evaluations[0]);

    update<<< p, 1 >>> (p_evaluations);
    cudaDeviceSynchronize();

    int sum = thrust::reduce(d_evaluations.begin(), d_evaluations.end(), 0, thrust::plus<int>());

    
    printf("%d\n", sum);
    


}
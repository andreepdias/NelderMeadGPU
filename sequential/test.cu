#include <iostream>


__global__ void update(int * e){

    (*e)++;

}


int main(){

    int * e;
    
    cudaMallocManaged(&e, sizeof(int));
    
    (*e) = 0;
    (*e)++;

    //printf("AAA\n");

    update<<< 10, 5 >>> (e);
    cudaDeviceSynchronize();

    printf("%d\n", (*e));
    


}
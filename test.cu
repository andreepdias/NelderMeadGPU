#include <bits/stdc++.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(){

    thrust::host_vector<float> h_vec(10);
    
    h_vec[0] = 0;
    h_vec[1] = 1;
    h_vec[2] = 2;
    h_vec[3] = 3;
    h_vec[4] = 4;
    h_vec[5] = 5;
    h_vec[6] = 10;
    h_vec[7] = 7;
    h_vec[8] = 8;
    h_vec[9] = 9;

    thrust::device_vector<float> d_vec = h_vec;
    float * p_vec = thrust::raw_pointer_cast(&d_vec[0]);

    thrust::device_vector<float>::iterator iter = thrust::max_element(p_vec, p_vec + 10);

    unsigned int position = iter - d_vec.begin();
    float max_val = *iter;

    std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;


    return 0;

}
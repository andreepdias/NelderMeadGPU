#include <bits/stdc++.h>
#include "src/gpu/util.cuh"
#include "src/gpu/objectiveFunctions.cuh"

int main(){


    std::ifstream file("test.txt");

    std::string aa;
    file >> aa;

    float x;
    std::vector<float> angles;

    while(file >> x){
        //angles.push_back(x * PI / 180.0f);
        angles.push_back(x);
    }    
    int dimension = angles.size();
    int protein_length = aa.size();
    
    NelderMead parameters;
    parameters.dimension = dimension;
    
    ABOffLattice * parametersAB = new ABOffLattice();
    (*parametersAB).protein_length = protein_length;
    (*parametersAB).aminoacid_sequence = aa.c_str();
    
    ABOffLattice * d_parametersAB;
    cudaMalloc(&d_parametersAB, sizeof(ABOffLattice));
    cudaMemcpy(d_parametersAB, parametersAB, sizeof(ABOffLattice), cudaMemcpyHostToDevice);
    
    char aa_sequence[150];
    memset(aa_sequence, 0, sizeof(char) * 150);
    strcpy(aa_sequence, (*parametersAB).aminoacid_sequence);
    cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));
    
	thrust::device_vector<float> d_simplex(dimension);	
    thrust::device_vector<uint>  d_indexes(1);
    thrust::device_vector<float> d_objective_function(1);
    
    float * p_angles = &angles[0];
	float * p_simplex			 		   = thrust::raw_pointer_cast(&d_simplex[0]);    
    float * p_objective_function 		   = thrust::raw_pointer_cast(&d_objective_function[0]);
    uint  * p_indexes 				       = thrust::raw_pointer_cast(&d_indexes[0]);
    
	thrust::copy(p_angles, p_angles + dimension, d_simplex.begin());
	thrust::sequence(d_indexes.begin(), d_indexes.end());

    calculateABOffLattice<<< 1, protein_length - 2 >>>(dimension, protein_length, p_simplex, p_objective_function);

    std::cout << d_objective_function[0] << std::endl;

    thrust::host_vector<float> h_simplex = d_simplex;
    printf("%2d. ", 4);
    for(int i = 0; i < dimension; i++){
        printf("%.5f ", h_simplex[i]);
    }
    printf("\n");

    return 0;

}
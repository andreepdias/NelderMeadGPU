#include <bits/stdc++.h>
#include "src/cpu/util.hpp"
#include "src/cpu/objectiveFunctions.hpp"


int main(){


    std::ifstream file("test.txt");

    std::string aa;
    file >> aa;

    float x;
    std::vector<float> angles;

    while(file >> x){
        angles.push_back(x * PI / 180.0f);
    }

    int dimension = angles.size();
    int protein_length = aa.size();

    NelderMead parameters;
    ABOffLattice * parametersAB = new ABOffLattice();

    parameters.dimension = dimension;

    (*parametersAB).protein_length = protein_length;
    (*parametersAB).aminoacid_sequence = aa.c_str();

    std::vector<std::pair<float, int> > obj_function(1);

    float * p_simplex = &angles[0];
    std::pair<float, int> * p_obj_function = &obj_function[0];

    calculateABOffLattice(parameters, (void*) parametersAB, 1, p_simplex, p_obj_function);

    std::cout << obj_function[0].first << std::endl;

    printf("%2d. ", 4);
    for(int i = 0; i < dimension; i++){
        printf("%.5f ", p_simplex[i]);
    }
    printf("\n");

    return 0;

}
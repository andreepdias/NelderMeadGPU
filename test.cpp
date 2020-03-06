#include <bits/stdc++.h>
#include "src/cpu/util.hpp"


int main(){


    std::ifstream file("test.txt");

    std::string aa;
    file >> aa;

    float x;
    std::vector<float> angles;

    while(file >> x){
        angles.push_back(x);
    }

    int dimension = angles.size();
    int protein_length = aa.size();

    NelderMead parameters;
    ABOffLattice * parametersAB = new ABOffLattice();

    parameters.dimension = dimension;

    (*parametersAB).protein_length = protein_length;
    (*parametersAB).aminoacid_sequence = aa.c_str();

    

    return 0;

}
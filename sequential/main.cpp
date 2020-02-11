#include "util.hpp"
#include "nelmin.hpp"


int main(){

    std::ifstream input_file("input.txt");
        
    std::string protein_name, protein_chain;
    std::vector<float> angles;

    input_file >> protein_name;
    input_file >> protein_chain;

    float angle;
    while(input_file >> angle){
        angles.push_back(angle * PI / 180.0f);
    }

    int protein_length = protein_chain.size();
    int dimension = angles.size();

    nelderMead(dimension, protein_length, &angles[0], protein_chain.c_str());

}
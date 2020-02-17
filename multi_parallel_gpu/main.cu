#include "util.h"
#include "nelmin.cuh"

int main(){

    /* Leitura do arquivo com nome da proteína, cadeia de aminoácidos e ângulos iniciais */
    std::ifstream input_file("input.txt");
        
    std::string protein_name, protein_chain;
    std::vector<float> angles;

    int iterations_number;

    input_file >> iterations_number;
    input_file >> protein_name;
    input_file >> protein_chain;

    float angle;
    while(input_file >> angle){
        angles.push_back(angle * PI / 180.0f);
        // angles.push_back(0.0f);
    }

    int protein_length = protein_chain.size();
    int dimension = angles.size();

    char aa_sequence[150];
    memset(aa_sequence, 0, sizeof(char) * 150);
    strcpy(aa_sequence, protein_chain.c_str());
    cudaMemcpyToSymbol(aminoacid_sequence, (void *) aa_sequence, 150 * sizeof(char));

    nelderMead(dimension, protein_length, &angles[0], iterations_number);

}
#include <bits/stdc++.h>
using namespace std;

#define PI 3.1415926535897932384626433832795029

float calculateABOffLattice(int dimension, int protein_length, const char * protein_chain, float * p_vertex){

    int number_evalueted_vertexes = 1;

	std::vector<float> aminoacid_position(protein_length * 3);

    int stride = 0;
    for(int i = 0; i < protein_length - 2; i++){

        aminoacid_position[0] = 0.0f;
        aminoacid_position[0 + protein_length] = 0.0f;
        aminoacid_position[0 + protein_length * 2] = 0.0f;

        aminoacid_position[1] = 0.0f;
        aminoacid_position[1 + protein_length] = 1.0f;
        aminoacid_position[1 + protein_length * 2] = 0.0f;

        aminoacid_position[2] = cosf(p_vertex[stride + 0]);
        aminoacid_position[2 + protein_length] = sinf(p_vertex[stride + 0]) + 1.0f;
        aminoacid_position[2 + protein_length * 2] = 0.0f;

        for(int j = 3; j < protein_length; j++){
            aminoacid_position[j] = aminoacid_position[j - 1] + cosf(p_vertex[stride + j - 2]) * cosf(p_vertex[stride + j + protein_length - 5]); // j - 3 + protein_length - 2
            aminoacid_position[j + protein_length] = aminoacid_position[j - 1 + protein_length] + sinf(p_vertex[stride + j - 2]) * cosf(p_vertex[stride + j + protein_length - 5]);
            aminoacid_position[j + protein_length * 2] = aminoacid_position[j - 1 + protein_length * 2] + sinf(p_vertex[stride + j + protein_length - 5]);
        }
    }

    float sum = 0.0f;

    for(int i = 0; i < protein_length - 2; i++){
        sum += (1.0f - cosf(p_vertex[stride + i])) / 4.0f;
    }

    float c, d, dx, dy, dz;

    for(int i = 0; i < protein_length - 2; i++){
        for(int j = i + 2; j < protein_length; j++){
            if(protein_chain[i] == 'A' && protein_chain[j] == 'A')
                c = 1.0;
            else if(protein_chain[i] == 'B' && protein_chain[j] == 'B')
                c = 0.5;
            else
                c = -0.5;

            dx = aminoacid_position[i] - aminoacid_position[j];
            dy = aminoacid_position[i + protein_length] - aminoacid_position[j + protein_length];
            dz = aminoacid_position[i + protein_length * 2] - aminoacid_position[j + protein_length * 2];
            d = sqrtf( (dx * dx) + (dy * dy) + (dz * dz) );

            sum += 4.0f * ( 1.0f / powf(d, 12.0f) - c / powf(d, 6.0f) );

        }
    }

    return sum;
}


struct Protein{

    int protein_length;
    int dimension;

    string protein_name;
    string protein_chain;

    vector<float> angles;

};

int main(){

    ifstream protein_file("resources/inputs/proteins.txt");

    if(!protein_file.is_open()){
        printf("File does not exist\n");
    }

    vector<Protein> proteins;
    float psl, n, x;
    string name, chain;

    while(protein_file >> name){
        Protein p;
        p.protein_name = name;
        protein_file >> p.protein_length >> p.dimension;
        protein_file >> p.protein_chain;
        
        for(int i  = 0; i < p.dimension; i++){
            protein_file >> x;
            p.angles.push_back(x * PI / 180.0f);
        }
        proteins.push_back(p);
    }

    printf("%d\n", proteins.size());

    printf("Printing:\n\n");

    for(int i = 0; i < proteins.size(); i++){
        printf("Protein Name: "); cout << proteins[i].protein_name << endl;
        printf("Length: %d, Dimension: %d\n", proteins[i].protein_length, proteins[i].dimension);
        printf("Protein Chain: "); cout << proteins[i].protein_chain << endl;;

        // for(int j = 0; j < proteins[i].angles.size(); j++){
            // printf("%.4f ", proteins[i].angles[j]);
        // }
        float obj = calculateABOffLattice(proteins[i].dimension, proteins[i].protein_length, proteins[i].protein_chain.c_str(), &proteins[i].angles[0]);
        printf("Obj Function: %.7f", obj);

        printf("\n\n");
    }
 }
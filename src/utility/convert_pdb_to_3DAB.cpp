#include <bits/stdc++.h>

#include "shared/abOffLattice.hpp"

char convertAminoacid(char &c){

    if(c == 'I' or c == 'V' or c == 'L' or c == 'P' or c == 'C' or c == 'M' or c == 'A'  or c == 'G'){
        return 'A';
    }
    else if(c == 'D' or c == 'E' or c == 'F' or c == 'H' or c == 'K' or c == 'N' or c == 'Q' or c == 'R' or c == 'S' or c == 'T' or c == 'W' or c == 'Y'){
        return 'B';
    }else{
        return 'X';
    }
}

int main(){

    std::vector<ABOffLattice> proteins;

    std::ifstream protein_file("proteins_pdb_raw.txt");

    std::string line, s, name, chain, mol;
    char c;
    int length;

    while(std::getline(protein_file, line)){
        std::stringstream ss(line);
        
        ss >> c;
        ss >> name;
        ss >> mol;        
        ss >> s;
        length = stoi(s.substr(s.find(":") + 1));

        std::getline(protein_file, chain);

        if(mol != "mol:protein"){
            continue;
        }

        bool unknown_aa = false;
        for(int i = 0; i < chain.length(); i++){
            c = convertAminoacid(chain[i]);

            if(c == 'X'){
                unknown_aa = true;
                break;
            }
            chain[i] = c;
        }
        if(unknown_aa){
           continue;
        }

        ABOffLattice p;
        p.protein_name = name;
        p.protein_length = length;
        p.aa_sequence = chain;
        p.aminoacid_sequence = p.aa_sequence.c_str();

        proteins.push_back(p);
    }

    auto comp = [](ABOffLattice &a, ABOffLattice &b){ return a.protein_length < b.protein_length; };

    std::sort(proteins.begin(), proteins.end(), comp);

    
    printf("Proteins count: %d\n\n", (int)proteins.size());

    for(int i = 0; i < (int)proteins.size(); i++){
        std::cout << proteins[i].protein_name << std::endl;
        std::cout << proteins[i].protein_length << std::endl;
        std::cout << proteins[i].aa_sequence << std::endl << std::endl;
    }
    
    return 0;
 }
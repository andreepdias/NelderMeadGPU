#include <bits/stdc++.h>

using namespace std;

void centroid(int d, int p){

    int n = d - p;

    


}

int main(){

    int dimension, processors;

    printf("Determine o numero de dimensoes: ");
    cin >> dimension;

    printf("Determine o numero de processadores: ");
    cin >> processors;
    
    int op;
    while(true){

        printf("\nEscolha a operacao (1- centroide, 2- reflexao, 3- expansao, 4- contracao, 5- encolhimento): ");
        cin >> op;

        switch(op){
            case 1:
                centroid(dimension, processors);
                break;
        }
    }

}
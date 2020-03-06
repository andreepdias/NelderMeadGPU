#include <bits/stdc++.h>

struct Protein{
    int size;
};


void do_nothing(Protein * & protein){

    protein = new Protein();
    (*protein).size = 69;

}

int main(){

    Protein * protein;

    do_nothing(protein);

    int s = (*protein).size;

    printf("%d\n", s);

    return 0;

}
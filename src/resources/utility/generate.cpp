#include <bits/stdc++.h>


int main(){

    std::random_device rd;
    std::mt19937 engine(rd());
    std::uniform_real_distribution<float> distribution(0, 1);

    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 200; j++){
            printf("%.7f ", distribution(engine));
        }
        printf("\n");
    }

}
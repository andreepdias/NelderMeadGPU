#include <bits/stdc++.h>


int round32(int x){
    return ((x + 31) / 32) * 32 ;
}

int main(){


    int x;

    while(true){

        std::cin >> x;

        std::cout << round32(x) << std::endl;

    }

}
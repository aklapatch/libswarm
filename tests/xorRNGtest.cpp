#include<iostream>
#define TIMES 10000

size_t rng(){
    static size_t x = 123456789;
    static size_t y = 362436069;
    static size_t z = 521288629;
    static size_t w = 88675123;
    size_t t;
    t = x ^ (x << 11);   
    x = y; 
    y = z; 
    z = w;   
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

int main(){
    for(int i = 0;++i<TIMES;)
        std::cout << "Random number: " << (float)(rng()%1000)/1000 << "\n";
    return 0;
}
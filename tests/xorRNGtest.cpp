#include<iostream>
#include<ctime>
#include<cstdlib>
#define TIMES 10

float rng(size_t x, size_t y){
	size_t z = x+y;
	size_t t = z ^ ( z << 11);
	size_t out = y ^ (y >> 19) ^ ( t ^ ( t >> 8));
	return (float)(out%1000)/1000;
}

int main(){
    srand(time(NULL));

    for(int i = 0;++i<TIMES;)
        std::cout << "random = " << rng(rand(),i) << "\n";
    return 0;
}
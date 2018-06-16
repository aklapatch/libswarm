#include <iostream>
#include <vector>

int test(std::vector<int> in){
    return in[0];
}

int main(){

    int test=32;
    std::vector<int> tvec(1);

    tvec[0]=test;
    
    std::cout << tvec[0] <<"\n";

    std::count << "Other test " << test({test, 1}) << "\n";

}

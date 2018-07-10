#include <iostream>
#include <vector>

int main(){

    int test=32;
    std::vector<int> tvec(3);
    tvec[2]=32;
    std::cout << "before = " << tvec[2] << "\n";
    tvec.data()[2]=60;
    std::cout << "after = " << tvec[2] << "\n";
}

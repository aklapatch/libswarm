///this code is being used in order to see how quick vector resizing is compared to
///dynamic * usage or certain methods of initialization
///some code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--

#include <iostream>
#include <vector>
#include <chrono>

int main(){

    std::vector<double> test (1);

    auto start =std::chrono::high_resolution_clock::now();

    test.resize(1000);

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish-start;

    std::cout << "Time for resize: " << elapsed.count() << " \n";

    test.resize(1);

    

    auto start =std::chrono::high_resolution_clock::now();

    test=gecvec(1000);

    auto finish = std::chrono::high_resolution_clock::now();

    elapsed = finish-start;

    std::cout << "Time for new allocation " << elapsed.count() << " \n";

}

std::vector<double> gecvec(int size){
    std::vector<double> ret (size);
    return ret;
}
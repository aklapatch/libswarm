///this code is being used in order to see how quick vector resizing is compared to
///dynamic * usage or certain methods of initialization
///some code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--

#include <iostream>
#include <vector>
#include <chrono>

std::vector<double> gecvec(int size){
    std::vector<double> ret (size);
    return ret;
}

#define SIZE 10000000

int main(){

    std::vector<double> test (1);

    auto start =std::chrono::high_resolution_clock::now();

    ///fastestish
    test.resize(SIZE);

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish-start;

    std::cout << "Time for resize: " << elapsed.count() << " \n";

    test.resize(1);

    start =std::chrono::high_resolution_clock::now();

    ///the slowest
    test=gecvec(SIZE);

    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish-start;

    std::cout << "Time for new allocation " << elapsed.count() << " \n";

    test.resize(1);

    start =std::chrono::high_resolution_clock::now();

    ///second slowest
    std::vector<double> temp (SIZE);
    test=temp;

    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish-start;

    std::cout << "Time for temp swap " << elapsed.count() << " \n";

    double * testptr= new double[SIZE];

    start =std::chrono::high_resolution_clock::now();

    ///the fastest by an lot of magnitude
    delete [] testptr;
    testptr= new double [SIZE];

    finish = std::chrono::high_resolution_clock::now();

    elapsed = finish-start;

    std::cout << "Time for double realloc " << elapsed.count() << " \n";

    delete [] testptr;


}


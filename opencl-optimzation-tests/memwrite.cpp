///an example to demonstrate usage of the opencl 1.2 C++ wrapper
/// code example usage https://github.com/Dakkers/OpenCL-examples/blob/master/example00/main.cpp
///timing code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--


#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
#include <chrono>
#include <vector>

#define ITEMS 1000

int main(){
    ///retrieve platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    ///get platfrom and print
    cl::Platform platform = platforms[0];
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    ///get devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    ///set devices
    cl::Device device=devices[0];
    cl::Device device1=devices[1];
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "Device1: " << device1.getInfo<CL_DEVICE_NAME>() << "\n";

    ///get context
    cl::Context context({device});

    ///make source and push it into source
    cl::Program::Sources sources;
    char  kersrc[]=
        "void kernel add (global int* A){"
        "int i=get_global_id(0);"
        "A[i]=362; }";
    sources.push_back({kersrc, sizeof(kersrc)});

    ///build program
    cl::Program program(context,sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    ///make queue
    cl::CommandQueue queue(context,device);

    ///make+init data
    int n=ITEMS,i=0;
    int * A = new int [n];
    for(i=0;i<n;++i){
        A[i]=0;
    }

    ///make buffer and write to it;
    cl::Buffer Abuf;
    Abuf=cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    queue.enqueueWriteBuffer(Abuf, CL_TRUE, 0, sizeof(int)*n, A);
    std::vector<cl::Event> wlist(1);

    cl::Kernel add=cl::Kernel(program,"add");

    ///measure time
    auto start = std::chrono::high_resolution_clock::now();

    ///set args
    add.setArg(0,Abuf);
    
    ///run kernel + read buffer answer
    queue.enqueueNDRangeKernel(add,cl::NullRange, cl::NDRange(n),cl::NullRange, NULL, &wlist[0]);
    cl::WaitForEvents(wlist);
    
    auto finish = std::chrono::high_resolution_clock::now();

    queue.enqueueReadBuffer(Abuf, CL_TRUE, 0, sizeof(int)*n,A, NULL,NULL);

    ///std::chrono::duration elapsed = finish - start;

    auto msec=std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start);

    std::cout <<"Time to execute value setting through kernel " << msec.count()<<"\n";

    start = std::chrono::high_resolution_clock::now();

        ///reset data
    for(i=0;i<n;++i){
        //std::cout << "i= " << i << "  A[i]= " << A[i] << "\n";
        A[i]=502;
    }
 
    queue.enqueueWriteBuffer(Abuf, CL_TRUE, 0, sizeof(int)*n, A,NULL,&wlist[0]);
    cl::WaitForEvents(wlist);

    finish = std::chrono::high_resolution_clock::now();

    msec=std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);

    std::cout <<"\nTime to execute write value to memory " << msec.count()<<"\n";

        ///reset data
    for(i=0;i<n;++i){
        //std::cout << "i= " << i << "  A[i]= " << A[i] << "\n";
    }

    return 0;
}
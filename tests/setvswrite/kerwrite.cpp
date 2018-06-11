///an example to demonstrate usage of the opencl 1.2 C++ wrapper
/// code example usage https://github.com/Dakkers/OpenCL-examples/blob/master/example00/main.cpp
///timing code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--


#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <iostream>
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif
#include <chrono>

#define ITEMS 100000

int main(){
    ///retrieve platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    ///get platfrom and print it
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
        "void kernel add (__global float * A){"
        "int i=get_global_id(0);"
        "A[i]=32;}";
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

    ///set args
    cl::Kernel add=cl::Kernel(program,"add");
    //add.setArg(0,Abuf);

    std::vector<cl::Event> vents(1);
    std::vector<cl::Event> ovent(1);

    ///measure time
    auto start = std::chrono::high_resolution_clock::now();

    add.setArg(0,Abuf);
    ///run kernel + read buffer answer
    queue.enqueueNDRangeKernel(add,cl::NullRange, cl:: NDRange(n),cl::NullRange,NULL,&vents[0]);
    cl::WaitForEvents(vents);

    auto finish = std::chrono::high_resolution_clock::now();

    ///std::chrono::duration elapsed = finish - start;

    auto msec=std::chrono::duration_cast<std::chrono::microseconds>(finish-start);

    std::cout <<"Time to execute " << msec.count()<<"\n";

    return 0;
}
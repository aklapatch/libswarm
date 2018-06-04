///an example to demonstrate usage of the opencl 1.2 C++ wrapper
/// code example usage https://github.com/Dakkers/OpenCL-examples/blob/master/example00/main.cpp
///timing code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--



#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>

#define ITEMS 100

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
	std::cout << "Device: spir extension " << device.getInfo<CL_DEVICE_SPIR_VERSIONS>() << "\n";
    std::cout << "Device1: spir " << device1.getInfo<CL_DEVICE_SPIR_VERSIONS>() << "\n";

    std::vector<cl::Device> devs(1);
    devs[0]=device;

    ///get context
    cl::Context context(devs);

    ///make source and push it into source
    cl::Program::Binaries binsrc;

    FILE * inf = fopen("test.bc","rb");

    fseek(inf,0,SEEK_END);

    size_t len= ftell(inf);

    char * kersrc = new char[len+1];

    std::cerr << "flength= " << len << "\n";

    rewind(inf);

   std::cerr << "flength= " << fread(kersrc,sizeof(char),len,inf) << "\n";

	kersrc[len]='\0';

    fclose(inf);

    binsrc.push_back({(const unsigned char *)kersrc, len+1});

    std::vector<cl_int> retvec(1);

    ///build program
    cl_int ret=0;
    cl::Program program(context, devs,binsrc,&retvec, &ret);
    std::cout << "ret= " << ret << "\n";

    if (program.build(devs,"-x spir -spir-std=1.2") != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devs[0],NULL) << std::endl;
        exit(1);
    }

    ///make queue
    cl::CommandQueue queue(context,device);

    ///make+init data
    cl_int n=ITEMS,i=ITEMS;
    cl_int * A = new int [n];
    while(i--) 
        A[i]=i;

    cl_float * result=NULL;    

    ///make buffer and write to it;
    cl::Buffer Abuf=cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * n);
    queue.enqueueWriteBuffer(Abuf, CL_TRUE, 0, sizeof(cl_int)*n, A);

    ///set args
    cl::Kernel add=cl::Kernel(program,"add");
    add.setArg(0,Abuf);

    ///run kernel + read buffer answer
    queue.enqueueNDRangeKernel(add,cl::NullRange, cl::NDRange(n),cl::NullRange );
    queue.enqueueReadBuffer(Abuf, CL_TRUE, 0, sizeof(cl_int)*n, A);

    delete [] kersrc;

    return 0;
}
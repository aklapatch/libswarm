///an example to demonstrate usage of the opencl 1.2 C++ wrapper
/// code example usage https://github.com/Dakkers/OpenCL-examples/blob/master/example00/main.cpp
///timing code from https://www.pluralsight.com/blog/software-development/how-to-measure-execution-time-intervals-in-c--



#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdio.h>

#define ITEMS 100

cl::Program::Binaries getbinary(const char * fname){
    std::ifstream inf(fname,std::ios::in | std::ios::binary);

    //get file size
    inf.seekg(0,std::ios::end);
    int len=inf.tellg();
    inf.seekg(0,std::ios::beg);

    std::vector<std::vector<unsigned char>> out(1);
    if(!inf){
        std::cerr << "file is not open\n";
        return out;
    }

    unsigned char tmp;
    for(int i=0;i<len;++i){
        inf >> tmp;
        out[0].push_back(tmp);
    }

    return out;
}

int main(){
    ///retrieve platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    ///get platfrom and print
    cl::Platform platform = platforms[0];
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

    ///get devices
    std::vector<cl::Device> dev;
    cl_int ret= platform.getDevices(CL_DEVICE_TYPE_GPU, &dev);
    std::cerr <<"ret= " << ret << "\n";

    ///get context
    cl::Context context(dev[0]);

    ///make source and push it into source
    cl::Program::Binaries bins=getbinary("test.clbin");

    //bins.push_back();

    std::vector<cl_int> retvec(1);

    ///build program
    cl::Program program=cl::Program(context, dev,bins,&retvec, &ret);
    std::cout << "ret= " << ret << " and revec[0] "<< retvec[0]<<"  \n";

    if (program.build(dev,NULL) != CL_SUCCESS) {
        std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev[0],NULL) << std::endl;
        exit(1);
    }


    ///make queue
    cl::CommandQueue queue(context,dev[0],NULL,&ret);

    cl::Buffer resbuf(context,CL_MEM_READ_WRITE,sizeof(float));
    cl::Kernel t(program,"t");
    std::cout << "arg set= " <<t.setArg(0,resbuf)<< "\n";

    queue.enqueueNDRangeKernel(t,cl::NullRange,cl::NDRange(1),cl::NullRange);

    cl_float result=0;
    queue.enqueueReadBuffer(resbuf,CL_TRUE,0,sizeof(cl_float),&result);

    std::cout << "Answer = " << result << std::endl;

    return 0;
}
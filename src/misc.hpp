#include "clSwarm.hpp"

template <typename T>
void printbuf(cl_mem buf, size_t size,cl_command_queue q){
	T * out = new T[size];

	int ret=clEnqueueReadBuffer(q, buf,CL_TRUE,0,size*sizeof(T),out);

	if(ret!=CL_SUCCESS)
		std::cerr << " Read error, code: " << ret << "\n";

	for (int i=-1;++i<size;)
		std::cout << "Buffer contents[" << i << "] = " << out[i] << "\n";

	delete [] out;
}

//checks opencl build and prints error message
void checkBuild(int errin, cl_program program, cl_device_id device);

//gets binary from program and writes it to filename
void writeBinary(cl_program prog,const char * filename);

//returns the binary file from the opened file handle
unsigned char * readBinary(FILE * fin, size_t * size);

//gets the platform with the most GPU's
cl_platform_id getclPlatform();

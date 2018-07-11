#include "clSwarm.hpp"

void checkBuild(int errin, cl_program program, cl_device_id device){
	if(errin!=CL_SUCCESS){
		char log[600];
		errin = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), (void *)log, NULL);
		std::cerr << "Build Failed.\nBuild Log:\n" << log << "\n";
		clReleaseProgram(program);
		clReleaseDevice(device);
		exit(1);
	}
}

//gets the platform with the most GPU's
cl_platform_id getclPlatform(){
	cl_uint numplats=0;
	int ret=CL_SUCCESS;
	ret = clGetPlatformIDs(10,NULL,&numplats);
	if(numplats == 0){
		std::cerr << "No opencl platforms found\n";
		exit(-1);
	}

	std::vector<cl_platform_id> tmp(numplats);

	ret = clGetPlatformIDs(numplats, tmp.data(),NULL);

	//finds platform with most GPU's
	cl_uint devnum=0, max=0, select=0;
	for(unsigned int i = 0; i < numplats; ++i){
		ret = clGetDeviceIDs(tmp[i],CL_DEVICE_TYPE_GPU,10,NULL,&devnum);
		if(devnum > max){
			select = i;
			max = devnum;
		}
	}

	//select platform and get device ids.
	return tmp[select];
}

//take cl program and write binary to file
void writeBinary(cl_program prog,const char * filename){
	
	size_t size=0, written=0;
	clGetProgramInfo(prog,CL_PROGRAM_BINARY_SIZES, sizeof(size_t),&size, &written);

	if(written>sizeof(size_t))
		std::cerr << "Number not written fully to argument\n";

	unsigned char * out = new unsigned char[size];

	clGetProgramInfo(prog,CL_PROGRAM_BINARIES,size,&out,&written);

	if(written > size)
		std::cerr << "binary not fully written to temp var.\n";
	
	std::ofstream fout(filename, std::ios::out | std::ios::binary);

	if(!fout){
		std::cerr << "Could not open file for writing.\n";
		exit(1);
	} 
	fout.write((const char *)out,sizeof(char)*size);
	fout.close();
	delete [] out;
}

unsigned char * readBinary(FILE * fin, size_t * size){
		
	fseek(fin,0,SEEK_END);
	*size = ftell(fin);
	unsigned char * binin = new unsigned char[*size];
	rewind(fin);
	fread(binin,sizeof(char),*size,fin);

	return binin;
}

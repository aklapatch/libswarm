// this code is taken from 
//https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/opencl-c/

#include <stdlib.h>			
#include <CL/cl.h>		
		
#include <stdio.h>		
		
int main() {		
	cl_platform_id platform_id = NULL;	
	cl_uint ret_num_platforms;	
	cl_device_id device_id = NULL;	
	cl_uint ret_num_devices;
	size_t local_size;
	size_t local_size_size;
		
	clGetPlatformIDs(1, &platform_id, &ret_num_platforms);	
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,	
		&ret_num_devices);
		
	clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_size), &local_size, &local_size_size);	
	printf("CL_DEVICE_LOCAL_MEM_SIZE = %d\n", (int)local_size);	
}
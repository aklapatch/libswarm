///this program is going to be used to find if it is faster to pass a float * to the kernel as an arg, or
///if you should allocate memory on the GPU instead.

#include <CL/cl.h>

int main() {
	///random data
	cl_float data[]={6,3,2,345,3,45,3,45,5,2,4,5,42,4,5};
	
	
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	
	cl_mem data = NULL;	
	cl_float result;
	
	cl_program program = NULL;
	cl_kernel kernel = NULL;	
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	
	///result data and input data
	cl_mem data = NULL;	
	cl_float result;
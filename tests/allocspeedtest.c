///this program is going to be used to find if it is faster to pass a float * to the kernel as an arg, or
///if you should allocate memory on the GPU instead.

#include <CL/cl.h>
#include<stdio.h>
#include<time.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define SIZE (0x100000)	
#define PRINTRET printf("ret= %d @ %d\n", ret, __LINE__);

int main() {
	//CPU data
	float data[]={6,3,2,345,3,45,3,45.3,5,2.5,4,5.6,42.3,4.2,5};
	cl_int num=15;
	float result=13.1;
	
	//GPU buffers
	cl_mem numbuf = NULL;	
	cl_mem databuf = NULL;	
	cl_mem resbuffer = NULL;
	
	//basic opencl information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;	
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret=0;	
	
	///open kernel
	FILE * fp = fopen("test.cl", "r");
	if(!fp){
		fprintf(stderr, "Failed to load kernel.\n");	
		exit(1);
	}	
	char * src=(char*)malloc(SIZE);
	size_t src_size=fread(src, 1, SIZE, fp);
	fclose(fp);

	PRINTRET;
	
	///get device info
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);	
	PRINTRET;
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	PRINTRET;
	///compile kernel and get context
	context = clCreateContext(NULL, 2, &device_id, NULL, NULL, &ret);
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	
	program = clCreateProgramWithSource(context, 1, (const char **)&src, (const size_t *)&src_size, &ret);
		
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	
	kernel = clCreateKernel(program, "add", &ret);
	
	///make result buffer
	resbuffer=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(command_queue, resbuffer, CL_TRUE, 0, sizeof(float), &result, 0, NULL, NULL);
	
	///write number data to gpu
	numbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(int), NULL, &ret);
	ret = clEnqueueWriteBuffer(command_queue, numbuf, CL_TRUE, 0, sizeof(int), &num, 0, NULL, NULL);
	
	//write data buffer to gpu
	databuf=clCreateBuffer(context, CL_MEM_READ_WRITE,num*sizeof(float), NULL, &ret);
	ret = clEnqueueWriteBuffer(command_queue, databuf, CL_TRUE, 0, num*sizeof(float), &data, 0, NULL, NULL);
	
	PRINTRET;
	
	///set kernel args
	ret=clSetKernelArg(kernel,0,sizeof(cl_mem), (void *)&resbuffer);
	PRINTRET;
	ret=clSetKernelArg(kernel,1,sizeof(cl_mem),&numbuf);
	PRINTRET;
	ret=clSetKernelArg(kernel,2,sizeof(cl_mem), (void *)&databuf);
	PRINTRET;
		
	///execute
	const size_t size=1;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,&size,NULL, 0,NULL,NULL);
	
	PRINTRET;
	
	printf("resultbefore= %f\n", result);
	
	//read back result
	ret = clEnqueueReadBuffer(command_queue, resbuffer, CL_TRUE, 0,sizeof(float), &result, 0, NULL, NULL);
	
	PRINTRET;
	
	printf("result= %f\n", result);
	
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);	
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(resbuffer);
	ret = clReleaseMemObject(databuf);
	ret = clReleaseMemObject(numbuf);	
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(src);
	
	return 0;
}	
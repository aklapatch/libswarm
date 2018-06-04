#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#include <stdbool.h>

#define len 500

void printDevInf(cl_device_id dev){
	int err=CL_SUCCESS;
	char msg[len];
	cl_uint iout=0;
	cl_bool bout=0;
	cl_ulong ulout=0;
	size_t tout=0;

	err=clGetDeviceInfo(dev,CL_DEVICE_NAME,len,msg,NULL);
	puts(msg);

	err=clGetDeviceInfo(dev,CL_DEVICE_ADDRESS_BITS,sizeof(iout),&iout,NULL);
	printf("CL_DEVICE_ADDRESS_BITS %d\n",iout);

	err=clGetDeviceInfo(dev,CL_DEVICE_COMPILER_AVAILABLE,sizeof(bout),&bout,NULL);
	printf("CL_DEVICE_COMPILER_AVAILABLE %d\n",bout);

	err=clGetDeviceInfo(dev,CL_DEVICE_ENDIAN_LITTLE,sizeof(bout),&bout,NULL);
	printf("CL_DEVICE_ENDIAN_LITTLE %d\n",bout);

	err=clGetDeviceInfo(dev,CL_DEVICE_EXTENSIONS,len, msg,NULL);
	printf("CL_DEVICE_EXTENSIONS err=%d %s\n",err,msg);

	err=clGetDeviceInfo(dev,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(cl_ulong),&ulout,NULL);
	printf("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE %llu\n",ulout);
}

int main(void){
	char msg[len];
	cl_int err;
	cl_platform_id * plat;
	cl_device_id devs;
	cl_uint plat_num=0,devnum=0;
	
	
	err=clGetPlatformIDs(1,NULL,&plat_num);
	
	plat = (cl_platform_id *)malloc(sizeof(cl_platform_id)*plat_num);
	
	err=clGetPlatformIDs(plat_num,plat,NULL);
	
	unsigned int i=plat_num;
	unsigned int j=0;
	while (i--){
	
		err=clGetPlatformInfo(plat[i],CL_PLATFORM_PROFILE,len,msg,NULL );
		
		puts(msg);
		
		err=clGetPlatformInfo(plat[i],CL_PLATFORM_VERSION,len,msg,NULL );
		
		puts(msg);
		
		err=clGetPlatformInfo(plat[i],CL_PLATFORM_NAME,len,msg,NULL );
		
		puts(msg);
		
		err=clGetPlatformInfo(plat[i],CL_PLATFORM_VENDOR,len,msg,NULL );
		
		puts(msg);
		
		err=clGetPlatformInfo(plat[i],CL_PLATFORM_EXTENSIONS,len,msg,NULL );
	
		printf("errr=%d\n",err);	
		puts(msg);	

		err=clGetDeviceIDs(plat[i],CL_DEVICE_TYPE_ALL,3,NULL,&devnum);

		err=clGetDeviceIDs(plat[i],CL_DEVICE_TYPE_ALL,devnum,&devs,NULL);

		printDevInf(devs);		

	}
	free(plat);
}

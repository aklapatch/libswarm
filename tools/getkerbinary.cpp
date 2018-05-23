///getkerbinary.c
/** this is a tool to print out the binary of a kernel source */

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fstream>
#define LOGSIZE 700
#define BUFF 1000



int main(int argc, char ** argv){

    //kernel length and kernel data
    unsigned int len=0,bsize=0, insize=0;
    char * txtin=NULL, blog[LOGSIZE];
    unsigned char * binout = NULL;
    FILE * file;
    std::ifstream infile;

    //opencl stuff
    cl_program program;
    cl_platform_id platform;
    cl_context ctx;
    cl_device_id dev_id;
    cl_int ret;

    //set up opencl environment
    ret=clGetPlatformIDs(1,&platform, NULL);
    ret=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&dev_id,NULL);
    ctx=clCreateContext(NULL,1,&dev_id,NULL,NULL,&ret);

    //cycle through all the arguments
    unsigned int i=argc;
    while(i-->1){
        
        infile.open(argv[i]);
        infile.seekg(0,std::ios::end);
        len=infile.tellg();
        infile.seekg(0,std::ios::beg);

        //get memory and read in file
        txtin=new char [len+1];
        infile.read(txtin,len);
        infile.close();
        txtin[len]='\0';
        
        printf("size= %d\n",len);

        puts(txtin);

        //build the program
        program = clCreateProgramWithSource(ctx,1,(const char **)&txtin,NULL, &ret);
        ret = clBuildProgram(program, 1,&dev_id,NULL,  NULL, NULL);

        //get the build log and print it
        ret = clGetProgramBuildInfo(program, dev_id, CL_PROGRAM_BUILD_LOG,LOGSIZE,blog,NULL);
        puts(blog);

        // get kernel binary
        ret = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,sizeof(unsigned int),&bsize,NULL);
        binout = (unsigned char *)malloc(sizeof(unsigned char)*bsize);
        ret = clGetProgramInfo(program, CL_PROGRAM_BINARIES,sizeof(unsigned char)*bsize,&binout,NULL);

        //openfile and output to file
        file=fopen(strcat(argv[i],"bin"),"wb");
        fwrite(binout,1,bsize,file);
        fclose(file);

        //free memory
        free(binout);
        delete [] txtin;
    }

    clReleaseProgram(program);
    clReleaseContext(ctx);
    clReleaseDevice(dev_id);

}
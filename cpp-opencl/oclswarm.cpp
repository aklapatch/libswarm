/*swarm.cpp
implementation for swarm
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "oclswarm.hpp"

#include <iostream>

///sets dimensions to 1 and number of particles to 100 and w to 1.0
swarm::swarm(){

    ///set swarm characteristics to defaults
    partnum=DEFAULT_PARTNUM;
    dimnum=DEFAULT_DIM;
    w = DEFAULT_W;
    c1=C1;
    c2=C2;
    gfitness=-HUGE_VAL;    

    ///gets up to 3 platforms
	ret = clGetPlatformIDs(PLATFORM_NUM, &platform_id, &ret_num_platforms);
	
	///gets up to 3 devices
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, DEVICE_NUM, &device_id, &ret_num_devices);
	
	///gets a context with up to 3 devices
	context = clCreateContext(NULL, ret_num_devices, &device_id, NULL, NULL, &ret);
	
	///get command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    ///use C++11 string literals to get kernel
    const char * src =
    #include "kernelstring.cl"
    ;
    
    ///build program
    program = clCreateProgramWithSource(context, 1, (const char **)&src, NULL, &ret);
		
	std::cout << "ret at 40 " << ret << "\n";
    
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    std::cout << "ret at 44 " << ret << "\n";
	
	distr = clCreateKernel(program, "distribute", &ret);

    cmpre= clCreateKernel(program, "compare", &ret);

    updte=clCreateKernel(program, "update", &ret);

    updte2=clCreateKernel(program, "update2", &ret);

    std::cout << "ret at 202 " << ret << "\n";

    ///create memory buffer fro gfit and write to it
    gfitbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, gfitbuf, CL_TRUE, 0, sizeof(cl_float), &gfitness, 0, NULL, NULL);

    ///create and write w to memory
    wbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, wbuf, CL_TRUE, 0, sizeof(cl_float), &w, 0, NULL, NULL);

    ///get memory for and write to c1buffer
    c1buf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, c1buf, CL_TRUE, 0, sizeof(cl_float), &c1buf, 0, NULL, NULL);

    ///get memory for and write to c2buffer
    c2buf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, c2buf, CL_TRUE, 0, sizeof(cl_float), &c2buf, 0, NULL, NULL);

    ///creates buffers for every resource
    presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);

    vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);

    cl_float * passin = new cl_float[partnum];

    ///make an array to pass -INF values to kernel
    unsigned int i;
    for(i=0;i<partnum;++i){
        passin[i]=-HUGE_VALF;
    }

    ///write values to memory
    pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, pfitnessbuf, CL_TRUE, 0, sizeof(cl_float), &passin, 0, NULL, NULL);

    delete [] passin;

    passin = new cl_float[dimnum];

    ///make an array to pass 0's to kernel
    for(i=0;i<dimnum;++i){
        passin[i]=0;
    }


    gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, gbestbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), &passin, 0, NULL, NULL);


    delete [] passin;

    passin = new cl_float[dimnum*partnum];

    ///make an array to pass 0's to kernel
    for(i=0;i<dimnum;++i){
        passin[i]=0;
    }

    pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, pbestbuf, CL_TRUE, 0, partnum*dimnum*sizeof(cl_float), &passin, 0, NULL, NULL);
}

///sets dimensions to 1 and number of particles to 100 and w to 1.5
swarm::swarm(unsigned int numdims, unsigned int numparts,cl_float inw){


    ///set swarm characteristics to defaults
    partnum=numparts;
    dimnum=numdims;
    w = inw;
    gfitness=-HUGE_VAL;    
    c1=C1;
    c2=C2;

    ///gets up to 3 platforms
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	
	///gets a  device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

	///gets a context with up to 3 devices
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	///get command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    ///use C++11 string literals to get kernel
    const char * src =
    #include "kernelstring.cl"
    ;
    
    ///build program
    program = clCreateProgramWithSource(context, 1, (const char **)&src, NULL, &ret);
    
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	
	distr = clCreateKernel(program, "distribute", &ret);

    cmpre= clCreateKernel(program, "compare", &ret);

    updte=clCreateKernel(program, "update", &ret);

    updte2=clCreateKernel(program, "update2", &ret);

    ///create memory buffer fro gfit and write to it
    gfitbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, gfitbuf, CL_TRUE, 0, sizeof(cl_float), &gfitness, 0, NULL, NULL);

    ///create and write w to memory
    wbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, wbuf, CL_TRUE, 0, sizeof(cl_float), &w, 0, NULL, NULL);

    ///get memory for and write to c1buffer
    c1buf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, c1buf, CL_TRUE, 0, sizeof(cl_float), &c1, 0, NULL, NULL);

    ///get memory for and write to c2buffer
    c2buf=clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, c2buf, CL_TRUE, 0, sizeof(cl_float), &c2, 0, NULL, NULL);

    ///creates buffers for every resource
    gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);

    presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);  

    cl_float * passin = new cl_float[partnum];

    ///make an array to pass -INF values to kernel
    unsigned int i;
    for(i=0;i<partnum;++i){
        passin[i]=-HUGE_VALF;
    }

    ///write values to memory
    pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, pfitnessbuf, CL_TRUE, 0, sizeof(cl_float), &passin, 0, NULL, NULL);

    delete [] passin;
}

///deallocate all memory
swarm::~swarm(){
    
    ///end queue
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
	
    ///get rid of kernels and program
	ret = clReleaseKernel(distr);
    ret = clReleaseKernel(updte);
    ret = clReleaseKernel(updte2);
    ret = clReleaseKernel(cmpre);
	ret = clReleaseProgram(program);

    ///release all used memory
    ret = clReleaseMemObject(fitnessbuf);
    ret = clReleaseMemObject(pfitnessbuf);
    ret = clReleaseMemObject(pbestbuf);
    ret = clReleaseMemObject(vbuf);
    ret = clReleaseMemObject(presentbuf);
    ret = clReleaseMemObject(wbuf);
    ret = clReleaseMemObject(c1buf);
    ret = clReleaseMemObject(c2buf);
    ret = clReleaseMemObject(gfitbuf);
    ret = clReleaseMemObject(gbestbuf);

    ///finish up queue and context
    ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}

///sets number of particles
void swarm::setpartnum(unsigned int num){
    
    ret = clReleaseMemObject(fitnessbuf);
    ret = clReleaseMemObject(pfitnessbuf);
    
    ret = clReleaseMemObject(pbestbuf);
    ret = clReleaseMemObject(vbuf);
    ret = clReleaseMemObject(presentbuf);

    ///reset particle swarm #
    partnum=num;

    ///does not need dimensions
    pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);

    fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float), NULL, &ret);  

    ///needs dimension to initialize
    presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);  
}

///sets number of dimensions
void swarm::setdimnum(unsigned int num){

    ///set dimension number    
    dimnum=num;

    ///free gpu memory
    ret = clReleaseMemObject(pbestbuf);
    ret = clReleaseMemObject(vbuf);
    ret = clReleaseMemObject(presentbuf);
    ret = clReleaseMemObject(gbestbuf);

    ///recreates buffers for every resource
    gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);

    presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);

    vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float), NULL, &ret);
}

///set inertial weight
void swarm::setweight(cl_float nw){

    ///create and write w to memory
    ret=clEnqueueWriteBuffer(command_queue, wbuf, CL_TRUE, 0, sizeof(cl_float), &nw, 0, NULL, NULL);
}

///set behavioral constants
void swarm::setconstants(cl_float nc1,cl_float nc2){
    ///write to c1buffer
    ret=clEnqueueWriteBuffer(command_queue, c1buf, CL_TRUE, 0, sizeof(cl_float), &nc1, 0, NULL, NULL);

    ///write to c2buffer
    ret=clEnqueueWriteBuffer(command_queue, c2buf, CL_TRUE, 0, sizeof(cl_float), &nc2, 0, NULL, NULL);

}

///distribute particle linearly from lower bound to upper bound
void swarm::distribute(cl_float * lower, cl_float * upper){
    
    ///make memory pool
    upperboundbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);
    lowerboundbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);

    ///store bounds for later
    ret=clEnqueueWriteBuffer(command_queue, upperboundbuf, CL_TRUE, 0, sizeof(cl_float), upper, 0, NULL, NULL);
    ret=clEnqueueWriteBuffer(command_queue, lowerboundbuf, CL_TRUE, 0, sizeof(cl_float), lower, 0, NULL, NULL);

    ///allocate memory for delta buffer
    cl_mem deltabuf= clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float), NULL, &ret);

    int oneless=partnum-1;

    partnumbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_uint), NULL, &ret);
    ret=clEnqueueWriteBuffer(command_queue, deltabuf, CL_TRUE, 0, sizeof(cl_uint), &oneless, 0, NULL, NULL);

    ///set kernel args
	ret=clSetKernelArg(distr,0,sizeof(cl_mem), (void *)&lowerboundbuf);
    ret=clSetKernelArg(distr,1,sizeof(cl_mem), (void *)&upperboundbuf);
    ret=clSetKernelArg(distr,2,sizeof(cl_mem), (void *)&deltabuf);
    ret=clSetKernelArg(distr,3,sizeof(cl_mem), (void *)&presentbuf);
    ret=clSetKernelArg(distr,4,sizeof(cl_mem), (void *)&pbestbuf);
    ret=clSetKernelArg(distr,5,sizeof(cl_mem), (void *)&partnumbuf);

    size_t gworksize[]={partnum,dimnum};

    ///execute
	ret = clEnqueueNDRangeKernel(command_queue, distr, 2, NULL,gworksize, NULL,0, NULL, NULL);
}

///run the position and velocity update equation
void swarm::update(unsigned int times){

    ///ready the parameters for the ndkernelrange
    size_t gworksize=partnum*dimnum;
    size_t * lworksize= new size_t[partnum];
    unsigned int i;
    for(i=0;i<partnum;++i){
        lworksize[i]=(size_t)dimnum;
    }

    ///make random number generator C++11
    std::random_device gen;
    std::uniform_real_distribution<float> distr(1,0);

    ///set up memory to take the random array
    cl_float * ran = new float [(1+dimnum)*partnum];
    cl_mem ranbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,(dimnum+1)*partnum*sizeof(float), NULL, &ret);

    while(times--){

        ///make a array of random numbers
        for(i=0;i<(dimnum+1)*partnum;++i){
            ran[i]= distr(gen);
        }

        ///write random numbers to buffer
        ret=clEnqueueWriteBuffer(command_queue, ranbuf, CL_TRUE, 0, (dimnum+1)*partnum*sizeof(cl_float), ran, 0, NULL, NULL);

        ///set kernel args
	    ret=clSetKernelArg(updte,0,sizeof(cl_mem), (void *)&presentbuf);
        ret=clSetKernelArg(updte,1,sizeof(cl_mem), (void *)&vbuf);
        ret=clSetKernelArg(updte,2,sizeof(cl_mem), (void *)&wbuf);
        ret=clSetKernelArg(updte,3,sizeof(cl_mem), (void *)&ranbuf);
        ret=clSetKernelArg(updte,4,sizeof(cl_mem), (void *)&pfitnessbuf);
        ret=clSetKernelArg(updte,5,sizeof(cl_mem), (void *)&upperboundbuf);
        ret=clSetKernelArg(updte,6,sizeof(cl_mem), (void *)&pbestbuf);
        ret=clSetKernelArg(updte,7,sizeof(cl_mem), (void *)&gbestbuf);
        ret=clSetKernelArg(updte,8,sizeof(cl_mem), (void *)&lowerboundbuf);
        ret=clSetKernelArg(updte,9,sizeof(cl_mem), (void *)&fitnessbuf);
        ret=clSetKernelArg(updte,10,sizeof(cl_mem), (void *)&partnumbuf);

        ///execute kernel
        ret = clEnqueueNDRangeKernel(command_queue, updte, 2, NULL,&gworksize,lworksize, 0,NULL,&ev);

        ///set args for compare kernel
        ret=clSetKernelArg(updte2,0,sizeof(cl_mem), (void *)&fitnessbuf);
        ret=clSetKernelArg(updte2,1,sizeof(cl_mem), (void *)&dimnumbuf);
        ret=clSetKernelArg(updte2,2,sizeof(cl_mem), (void *)&pfitnessbuf);
        ret=clSetKernelArg(updte2,3,sizeof(cl_mem), (void *)&presentbuf);
        ret=clSetKernelArg(updte2,4,sizeof(cl_mem), (void *)&pbestbuf);
        ret=clSetKernelArg(updte2,5,sizeof(cl_mem), (void *)&partnumbuf);

        ret= clEnqueueNDRangeKernel(command_queue, updte2,1,NULL,(const size_t*)&partnum,NULL,1, &ev,NULL);

        ///set kernel args
	    ret=clSetKernelArg(updte,0,sizeof(cl_mem), (void *)&presentbuf);
        ret=clSetKernelArg(updte,1,sizeof(cl_mem), (void *)&gbestbuf);
        ret=clSetKernelArg(updte,2,sizeof(cl_mem), (void *)&fitnessbuf);
        ret=clSetKernelArg(updte,3,sizeof(cl_mem), (void *)&gfitness);
        ret=clSetKernelArg(updte,4,sizeof(cl_mem), (void *)&partnumbuf);
        ret=clSetKernelArg(updte,5,sizeof(cl_mem), (void *)&dimnumbuf);

        ret= clEnqueueTask(command_queue, cmpre,1,&ev,NULL);
    }    

    ///release the buffer
    ret = clReleaseMemObject(ranbuf);

    delete [] ran;
    delete [] lworksize;
}

///returns best position of the swarm
cl_float * swarm::getgbest(){
    cl_float * gbest = new cl_float[dimnum];

    ///get value from buffer
    ret = clEnqueueReadBuffer(command_queue, gbestbuf, CL_TRUE, 0,dimnum*sizeof(cl_float), gbest, 0, NULL, NULL);
	
    return gbest;
}   

///returns the fitness of the best particle
cl_float swarm::getgfitness(){

    ///get value from GPU
    ret = clEnqueueReadBuffer(command_queue, gfitbuf, CL_TRUE, 0,sizeof(cl_float), &gfitness, 0, NULL, NULL);

    return gfitness;
}

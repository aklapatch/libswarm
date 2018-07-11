/*clSwarm.cpp
implementation for swarm
for opencl version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "clSwarm.hpp"
#include <iostream>

cl_float * getarray(size_t size, cl_float value){
	cl_float * out=new cl_float[size];
	while(size--)
		out[size]=value;

	return out;
}

void checkBuild(int errin, cl_program program, cl_device_id device){
	if(errin!=CL_SUCCESS){
		char log[600];
		errin = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 600, (void *)log, NULL);
		std::cerr << "Build Failed.\nBuild Log:\n" << log << "\n";
		exit(1);
	}
}

void writeBinary(cl_program prog,const char * filename){
	
	size_t size=0, written=0;
	clGetProgramInfo(prog,CL_PROGRAM_BINARY_SIZES, sizeof(size_t),&size, &written);

	if(written>sizeof(size_t))
		std::cerr << "Number not written fully to argument\n";

	unsigned char * out = new unsigned char[size];

	clGetProgramInfo(prog,CL_PROGRAM_BINARIES,size,out,&written);

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

//sets dimensions to 1 and number of particles to 100 and w to 1.0
clSwarm::clSwarm() {

	//set swarm characteristics to defaults
	partnum=DEFAULT_PARTNUM;
	dimnum=DEFAULT_DIM;
	w = DEFAULT_W;
	c1 = DEFAULT_C1;
	c2 = DEFAULT_C2;

	//gets platforms
	ret = clGetPlatformIDs(1, &platform,NULL);

	//finds gpus
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//gets a context with the first GPU found devices
	context=clCreateContext(NULL , 1, &device, NULL, NULL, &ret);

	//get command queue
	queue=clCreateCommandQueueWithProperties(context, device, NULL, &ret);

	//use C++11 string literals to get kernel
	const char src[]= R"(
		//kernels.cl
/** houses all kernels for this project */

//include fitness function
#include "fitness.cl"

//return the index with the biggest number
int sort(__global float * array,unsigned int size){
	int out=0;
	unsigned int i = size;
	float biggest=-INFINITY;
	while(i-->0){
		if(array[i]>biggest){
			biggest=array[i];
			out=i;
		}
	}
	return out;
}

__kernel void compare( __global float *presents,
						__global float * gbest,
						__global float * fitnesses,
						__global float * gfitness,
						unsigned int partnum,
						unsigned int dimnum) {

	//copy most fit particle into the gbest array
	unsigned int i=dimnum;
	unsigned int index=sort(fitnesses,partnum);

	if(fitnesses[index] > *gfitness) {

		//copy the array fitness
		*gfitness=fitnesses[index];

		//copy array into gbest array
		while(i--) 
			gbest[i]=presents[index*dimnum+i];
	}
}

//TEST FUNCTION
//get the delta array
__kernel void getDelta(__global float * lowerbound,
					__global float * upperbound,
					__global float * delta,
					unsigned int partnum) {
	//compute the delta
	unsigned int i=get_global_id(0);
	delta[i]=(upperbound[i]-lowerbound[i])/(partnum - 1);
}

//TEST FUNCTION
__kernel void distrtest(__global float * lowerbound,
						__global float * delta,
						__global float * presents,
						__global float * pbests,
						unsigned int dimnum,
						unsigned int partnum){

	//get_global_id(1) is dimension number, get_global_id(0) is particle number
	unsigned int i[2]={get_global_id(1), get_global_id(0)*dimnum + get_global_id(1)};

	//does the distribution sets pbests=0
	presents[i[1]]=get_global_id(0)*delta[i[0]] + lowerbound[i[0]];
}

//distributes particles linearly between the bounds
__kernel void distribute(__global float * lowerbound,
						 __global float * upperbound,
						 __global float * presents,
						 unsigned int dimnum,
						 unsigned int partnum){
		//get_global_id(1) is dimension number, get_global_id(0) is particle number					 
	uint part=get_global_id(0);	
	uint dim = get_global_id(1);
	uint pdex = part*dimnum + get_global_id(1);

	//distribute the particle between the upper and lower boundaries linearly
	presents[pdex]=part*((upperbound[dim]-lowerbound[dim])/(partnum - 1)) + lowerbound[dim];
}

float rng(uint x, uint y){
	uint z = x+y;
	uint t = z ^ ( z << 11);
	uint out = y ^ (y >> 19) ^ ( t ^ ( t >> 8));
	return (float)(out%1000)/1000;
}

__kernel void update( __global float * presents,
					  __global float * v,
					  float w,
					  unsigned int seed,
					  __global float * pfitnesses,
					  __constant float *upperbound,
					  __global float * pbest,
					  __global float * gbest,
					  __constant float * lowerbound,
					  __global float * fitnesses,
						unsigned int dimnum,
					  float c1,
					  float c2) {
					  
	uint index= get_global_id(0)*dimnum + get_global_id(1);
	uint dex0=get_global_id(0);
	uint dex1=get_global_id(1);

	//get_global_id(0) is partnum, get_global_id(1) is dimiension number
	//velocity update
	v[index]=w*v[index]
	 + c1*rng(seed, index)*(pbest[index]- presents[index])
	 + c2*rng(seed, index)*(gbest[dex1]-presents[index]);

	//position update
	presents[index]=presents[index]+v[index];

	//upper bound check
	if(presents[index]>upperbound[dex1]){
		presents[index]=upperbound[dex1];

	//lower bounds check
	} else if(presents[index]<lowerbound[dex1]){
		presents[index]=lowerbound[dex1];
	}
}

//compares and copies a coordinates into a pbest if necessary
__kernel void update2(__global float * fitnesses,
						unsigned int dimnum,
						__global float * pfitnesses,
						__global float * presents,
						__global float * pbest,
						unsigned int partnum) {

	const unsigned int j=get_global_id(0);
	unsigned int offset=j*dimnum;

	//evaluate fitness of the particle
	/** fitness function is in fitness.cl */
	fitnesses[j]=fitness(presents,offset, dimnum);

	//if the fitness is better than the pfitness, copy the values to pbest array
	if(fitnesses[j]>pfitnesses[j]) {

		//copy new fitness
		pfitnesses[j]=fitnesses[j];

		unsigned int i=dimnum;

		while(i--)
			pbest[offset+i]=presents[offset+i];
	}
}

	)";

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";

		size_t srcsize = sizeof(src);
		program = clCreateProgramWithSource(context, 1, (const char **)&src, &srcsize, &ret);
		ret = clBuildProgram(program, 1, &device, " ", NULL, NULL);
		checkBuild(ret,program,device);

		writeBinary(program,"kernels.bin");

	//if there is a binary, get it and use it
	} else {
		size_t size[1] ;
		unsigned char * tmpbin = readBinary(binaryfile, size);
		
		program= clCreateProgramWithBinary(context,1,&device, size, (const unsigned char **)&(tmpbin), NULL,&ret);

		delete [] tmpbin;

		ret = clBuildProgram(program, 1, &device, " -cl-std=CL2.1 ", NULL, NULL);

		checkBuild(ret,program,device);
	}

	//get kernels
	distr = clCreateKernel(program, "distribute", &ret);
	cmpre= clCreateKernel(program, "compare",&ret);
	updte=clCreateKernel(program, "update",&ret);
	updte2=clCreateKernel(program, "update2",&ret);

	//create buffers for particle positions, velocities, velocities, fitnesses
	presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	std::vector<float> tmp(partnum*dimnum,0);

	//set buffers for gbests, and pbests to 0
	gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//set the vectors values
	for(int dex =-1;++dex<partnum;)
		tmp[dex]=-HUGE_VALF;

	//create memory buffer for nonparticle fitnesses
	gfitbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL,&ret);
	ret= clEnqueueWriteBuffer(queue, gfitbuf, CL_TRUE , 0, sizeof(cl_float),tmp.data(),0,NULL,&ev);
	evs.emplace_back(ev);
	pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float) ,NULL,&ret);
	ret= clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),0,NULL,&ev);
	evs.emplace_back(ev);

	//make memory pool for upper and lower bounds
	upperboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
}

//sets all properties according to arguments
clSwarm::clSwarm(cl_uint numparts, cl_uint numdims,cl_float inw, cl_float c1in, cl_float c2in){

	//set properties
	partnum=numparts;
	dimnum=numdims;
	w=inw;
	c1=c1in;
	c2=c2in;

	//gets platforms
	ret = clGetPlatformIDs(1, &platform,NULL);

	//finds gpus
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//gets a context with the first GPU found devices
	context=clCreateContext(NULL , 1, &device, NULL, NULL, &ret);

	//get command queue
	queue=clCreateCommandQueueWithProperties(context, device, NULL, &ret);

	//use C++11 string literals to get kernel
	std::string src{R"(
		
#include "fitness.cl"

//return the index with the biggest number
int sort(__global float * array,unsigned int size){
	int out=0;
	unsigned int i = size;
	float biggest=-INFINITY;
	while(i-->0){
		if(array[i]>biggest){
			biggest=array[i];
			out=i;
		}
	}
	return out;
}

__kernel void compare( __global float *presents,
						__global float * gbest,
						__global float * fitnesses,
						__global float * gfitness,
						unsigned int partnum,
						unsigned int dimnum) {

	//copy most fit particle into the gbest array
	unsigned int i=dimnum;
	unsigned int index=sort(fitnesses,partnum);

	if(fitnesses[index] > *gfitness) {

		//copy the array fitness
		*gfitness=fitnesses[index];

		//copy array into gbest array
		while(i--) 
			gbest[i]=presents[index*dimnum+i];
	}
}

//TEST FUNCTION
//get the delta array
__kernel void getDelta(__global float * lowerbound,
					__global float * upperbound,
					__global float * delta,
					unsigned int partnum) {
	//compute the delta
	unsigned int i=get_global_id(0);
	delta[i]=(upperbound[i]-lowerbound[i])/(partnum - 1);
}

//TEST FUNCTION
__kernel void distrtest(__global float * lowerbound,
						__global float * delta,
						__global float * presents,
						__global float * pbests,
						unsigned int dimnum,
						unsigned int partnum){

	//get_global_id(1) is dimension number, get_global_id(0) is particle number
	unsigned int i[2]={get_global_id(1), get_global_id(0)*dimnum + get_global_id(1)};

	//does the distribution sets pbests=0
	presents[i[1]]=get_global_id(0)*delta[i[0]] + lowerbound[i[0]];
}

//distributes particles linearly between the bounds
__kernel void distribute(__global float * lowerbound,
						 __global float * upperbound,
						 __global float * presents,
						 unsigned int dimnum,
						 unsigned int partnum){
		//get_global_id(1) is dimension number, get_global_id(0) is particle number					 
	uint part=get_global_id(0);	
	uint dim = get_global_id(1);
	uint pdex = part*dimnum + get_global_id(1);

	//distribute the particle between the upper and lower boundaries linearly
	presents[pdex]=part*((upperbound[dim]-lowerbound[dim])/(partnum - 1)) + lowerbound[dim];
}

float rng(uint x, uint y){
	uint z = x+y;
	uint t = z ^ ( z << 11);
	uint out = y ^ (y >> 19) ^ ( t ^ ( t >> 8));
	return (float)(out%1000)/1000;
}

__kernel void update( __global float * presents,
					  __global float * v,
					  float w,
					  unsigned int seed,
					  __global float * pfitnesses,
					  __constant float *upperbound,
					  __global float * pbest,
					  __global float * gbest,
					  __constant float * lowerbound,
					  __global float * fitnesses,
						unsigned int dimnum,
					  float c1,
					  float c2) {
					  
	uint index= get_global_id(0)*dimnum + get_global_id(1);
	uint dex0=get_global_id(0);
	uint dex1=get_global_id(1);

	//get_global_id(0) is partnum, get_global_id(1) is dimiension number
	//velocity update
	v[index]=w*v[index]
	 + c1*rng(seed, index)*(pbest[index]- presents[index])
	 + c2*rng(seed, index)*(gbest[dex1]-presents[index]);

	//position update
	presents[index]=presents[index]+v[index];

	//upper bound check
	if(presents[index]>upperbound[dex1]){
		presents[index]=upperbound[dex1];

	//lower bounds check
	} else if(presents[index]<lowerbound[dex1]){
		presents[index]=lowerbound[dex1];
	}
}

//compares and copies a coordinates into a pbest if necessary
__kernel void update2(__global float * fitnesses,
						unsigned int dimnum,
						__global float * pfitnesses,
						__global float * presents,
						__global float * pbest,
						unsigned int partnum) {

	const unsigned int j=get_global_id(0);
	unsigned int offset=j*dimnum;

	//evaluate fitness of the particle
	/** fitness function is in fitness.cl */
	fitnesses[j]=fitness(presents,offset, dimnum);

	//if the fitness is better than the pfitness, copy the values to pbest array
	if(fitnesses[j]>pfitnesses[j]) {

		//copy new fitness
		pfitnesses[j]=fitnesses[j];

		unsigned int i=dimnum;

		while(i--)
			pbest[offset+i]=presents[offset+i];
	}
}
   )"};

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";
		//fclose(binaryfile);

		size_t srcsize = src.size();
		const char * tmpsrc = src.data();
		program = clCreateProgramWithSource(context, 1, (const char **)&tmpsrc, &srcsize, &ret);
		ret = clBuildProgram(program, 1, &device, " ", NULL, NULL);
		checkBuild(ret,program,device);

		writeBinary(program,"kernels.bin");

	//if there is a binary, get it and use it
	} else {
		size_t size[1] ;
		unsigned char * tmpbin = readBinary(binaryfile, size);
		fclose(binaryfile);
		program= clCreateProgramWithBinary(context,1,&device, size, (const unsigned char **)&(tmpbin), NULL,&ret);

		delete [] tmpbin;

		ret = clBuildProgram(program, 1, &device, "", NULL, NULL);

		checkBuild(ret,program,device);
	}

	//get kernels
	distr = clCreateKernel(program, "distribute", &ret);
	cmpre= clCreateKernel(program, "compare",&ret);
	updte=clCreateKernel(program, "update",&ret);
	updte2=clCreateKernel(program, "update2",&ret);

	//create buffers for particle positions, velocities, velocities, fitnesses
	presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	std::vector<float> tmp(partnum*dimnum,0);

	//set buffers for gbests, and pbests to 0
	gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//set the vectors values
	for(int dex =-1;++dex<partnum;)
		tmp[dex]=-HUGE_VALF;

	//create memory buffer for nonparticle fitnesses
	gfitbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL,&ret);
	ret= clEnqueueWriteBuffer(queue, gfitbuf, CL_TRUE , 0, sizeof(cl_float),tmp.data(),0,NULL,&ev);
	evs.emplace_back(ev);
	pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float) ,NULL,&ret);
	ret= clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),0,NULL,&ev);
	evs.emplace_back(ev);

	//make memory pool for upper and lower bounds
	upperboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
}

//the destructor
clSwarm::~clSwarm(){

	//finish and flush everything
	ret = clFlush(queue);
	ret = clFinish(queue);
	ret = clReleaseContext(context);
	ret= clReleaseCommandQueue(queue);
}

//sets number of particles
void clSwarm::setPartNum(cl_uint num){

	//reset particle swarm #
	partnum=num;

	//needs dimension to initialize
	presentbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	pbestbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	vbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	std::vector<cl_float> tmp(partnum,-HUGE_VALF);

	//does not need dimensions
	pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	ret = clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),evs.size(),evs.data(),&ev);
	evs.emplace_back(ev);

	fitnessbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	//re-writes values to gfitness
	ret = clEnqueueWriteBuffer(queue, gfitbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),0,NULL,&ev);
	evs.emplace_back(ev);
}

//returns particle number
cl_uint clSwarm::getPartNum(){
	return partnum;
}

//sets number of dimensions
void clSwarm::setDimNum(cl_uint num){

	//set dimension number
	dimnum=num;

	//recreates buffers for every resource
	gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);

	presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//make memory pool for upper and lower bounds
	upperboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);

	std::vector<cl_float> tmp(partnum,-HUGE_VALF);

	ret= clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(), evs.size(), evs.data(), &ev);
	evs.emplace_back(ev);

	//re-writes values to gfitness
	ret=clEnqueueWriteBuffer(queue, gfitbuf,CL_TRUE,0, sizeof(cl_float),tmp.data(),0, NULL,&ev);
	evs.emplace_back(ev);
}

//returns dimension number
cl_uint clSwarm::getDimNum(){
	return dimnum;
}

//set inertial weight
void clSwarm::setWeight(cl_float inw){
	w=inw;	
}

//return inertial weight
cl_float clSwarm::getWeight(){
	return w;
}

//set behavioral constants
void clSwarm::setC1(cl_float inc1){
	c1=inc1;
}

//set behavioral constants
void clSwarm::setC2(cl_float inc2){
	c2=inc2;
}

//return constant
cl_float clSwarm::getC1(){
	return c1;
}

//return constant
cl_float clSwarm::getC2(){
	return c2;
}

//distribute particle linearly from lower bound to upper bound
void clSwarm::distribute(cl_float * lower, cl_float * upper){

	//store bounds for later
	ret= clEnqueueWriteBuffer(queue, upperboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), upper,evs.size(), evs.data(),&ev);
	evs.emplace_back(ev);
	
	ret=clEnqueueWriteBuffer(queue, lowerboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), lower,0,NULL,&ev);
	evs.emplace_back(ev);

	//set kernel args
	ret = clSetKernelArg(distr, 0, dimnum*sizeof(cl_float), lowerboundbuf);
	ret = clSetKernelArg(distr,1, dimnum*sizeof(cl_float), upperboundbuf);
	ret = clSetKernelArg(distr,2, dimnum*partnum*sizeof(cl_float), presentbuf);
	ret = clSetKernelArg(distr,3, sizeof(cl_uint), &dimnum);
	ret = clSetKernelArg(distr,4, sizeof(cl_uint), &partnum);

	//set up work dimensionts
	std::vector<size_t> dim = {partnum, dimnum};

	//execute
	ret = clEnqueueNDRangeKernel(queue, distr, 2, NULL, dim.data(), NULL ,0 ,NULL,&ev);
	evs.emplace_back(ev);
}

//run the position and velocity update equation
void clSwarm::update(unsigned int times){

	//seed random to use during process
	srand(time(NULL));

	std::vector<size_t> dim = {partnum, dimnum};

	// updates everything
	while(times--){

		//set args for fitness eval
		ret = clSetKernelArg(updte2,0, partnum*sizeof(cl_uint), fitnessbuf);
		ret = clSetKernelArg(updte2,1, sizeof(cl_uint), &dimnum);
		ret = clSetKernelArg(updte2,2, partnum*sizeof(cl_float) ,  pfitnessbuf);
		ret = clSetKernelArg(updte2,3, dimnum*partnum*sizeof(cl_float), presentbuf);
		ret = clSetKernelArg(updte2,4, dimnum*partnum*sizeof(cl_float), pbestbuf);
		ret = clSetKernelArg(updte2,5, sizeof(cl_uint), &partnum);

		ret= clEnqueueNDRangeKernel(queue, updte2, 1, NULL, (size_t*)&partnum,NULL, 0, NULL, &ev);
		evs.emplace_back(ev);

		//set kernel args
		ret=clSetKernelArg(cmpre,0, dimnum*partnum*sizeof(cl_float), presentbuf);
		ret=clSetKernelArg(cmpre,1, dimnum*sizeof(cl_float), gbestbuf);
		ret=clSetKernelArg(cmpre,2, partnum*sizeof(cl_float), fitnessbuf);
		ret=clSetKernelArg(cmpre,3, sizeof(cl_float), gfitbuf);
		ret=clSetKernelArg(cmpre,4, sizeof(cl_uint), &partnum);
		ret=clSetKernelArg(cmpre,5, sizeof(cl_uint), &dimnum);

		//wait then run comparison
		size_t one = 1;
		ret= clEnqueueNDRangeKernel(queue, cmpre, 1,NULL, &one, NULL ,0, NULL, &ev);
		evs.emplace_back(ev);

		//set kernel args
		cl_uint seed = rand();
		ret=clSetKernelArg(updte,0, partnum*dimnum*sizeof(cl_float), presentbuf);
		ret=clSetKernelArg(updte,1, partnum*dimnum*sizeof(cl_float), vbuf);
		ret=clSetKernelArg(updte,2, sizeof(cl_float), &w);
		ret=clSetKernelArg(updte,3, sizeof(cl_uint), &seed);
		ret=clSetKernelArg(updte,4, partnum*sizeof(cl_float), pfitnessbuf);
		ret=clSetKernelArg(updte,5, dimnum*sizeof(cl_float), upperboundbuf);
		ret=clSetKernelArg(updte,6, partnum*dimnum*sizeof(cl_float), pbestbuf);
		ret=clSetKernelArg(updte,7, dimnum*sizeof(cl_float), gbestbuf);
		ret=clSetKernelArg(updte,8, dimnum*sizeof(cl_float), lowerboundbuf);
		ret=clSetKernelArg(updte,9, partnum*sizeof(cl_float), fitnessbuf);
		ret=clSetKernelArg(updte,10, sizeof(cl_uint), &dimnum);
		ret=clSetKernelArg(updte,11, sizeof(cl_float), &c1);
		ret=clSetKernelArg(updte,12, sizeof(cl_float), &c2);

		//wait then execute
		ret=clEnqueueNDRangeKernel(queue,updte, 2, NULL ,  (const size_t *)dim.data() , NULL,0, NULL, &ev);
		evs.emplace_back(ev);
	}

	//set args for fitness eval
	ret = clSetKernelArg(updte2,0, partnum*sizeof(cl_uint), fitnessbuf);
	ret = clSetKernelArg(updte2,1, sizeof(cl_uint), &dimnum);
	ret = clSetKernelArg(updte2,2, partnum*sizeof(cl_float) ,  pfitnessbuf);
	ret = clSetKernelArg(updte2,3, dimnum*partnum*sizeof(cl_float), presentbuf);
	ret = clSetKernelArg(updte2,4, dimnum*partnum*sizeof(cl_float), pbestbuf);
	ret = clSetKernelArg(updte2,5, sizeof(cl_uint), &partnum);

	ret= clEnqueueNDRangeKernel(queue, updte2, 1, NULL, (const size_t *)&partnum,NULL, 0, NULL, &ev);
	evs.emplace_back(ev);

	//set kernel args
	ret=clSetKernelArg(cmpre,0, dimnum*partnum*sizeof(cl_float), presentbuf);
	ret=clSetKernelArg(cmpre,1, dimnum*sizeof(cl_float), gbestbuf);
	ret=clSetKernelArg(cmpre,2, partnum*sizeof(cl_float), fitnessbuf);
	ret=clSetKernelArg(cmpre,3, sizeof(cl_float), gfitbuf);
	ret=clSetKernelArg(cmpre,4, sizeof(cl_uint), &partnum);
	ret=clSetKernelArg(cmpre,5, sizeof(cl_uint), &dimnum);

	//wait then run comparison
	size_t one = 1;
	ret= clEnqueueNDRangeKernel(queue, cmpre, 1,NULL, &one, NULL ,0, NULL, &ev);
	evs.emplace_back(ev);
}

//sets particle data
void clSwarm::setPartData(cl_float * in){
	ret = clEnqueueWriteBuffer(queue, presentbuf,CL_TRUE, 0,partnum*dimnum*sizeof(cl_float),in,evs.size() , evs.data(), &ev);
	evs.emplace_back(ev);
}

//copies particle data to the argument
void clSwarm::getPartData(cl_float * out){
	ret = clEnqueueReadBuffer(queue, presentbuf,CL_TRUE,0,partnum*dimnum*sizeof(cl_float),out,evs.size(), evs.data(), &ev);
	evs.emplace_back(ev);
}

//returns the fitness of the best particle
cl_float clSwarm::getGFitness(){

	cl_float out;

	//get value from GPU
	ret=clEnqueueReadBuffer(queue, gfitbuf, CL_TRUE, 0,sizeof(cl_float), &out, evs.size(), evs.data() ,&ev);
	evs.emplace_back(ev);

	return out;
}

//returns best position of the clSwarm
void clSwarm::getGBest(cl_float * out){

	//get value from buffer
	ret= clEnqueueReadBuffer(queue, gbestbuf, CL_TRUE, 0,dimnum*sizeof(cl_float),out, evs.size(), evs.data(), &ev);
	evs.emplace_back(ev);
}

void clSwarm::wait(){
	clEnqueueBarrierWithWaitList(queue, evs.size(), evs.data(), &ev);
}

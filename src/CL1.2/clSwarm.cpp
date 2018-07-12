/*clSwarm.cpp
implementation for swarm
for opencl version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "clSwarm.hpp"
#include "misc.cpp"

//sets dimensions to 1 and number of particles to 100 and w to 1.0
clSwarm::clSwarm() {

	//set swarm characteristics to defaults
	partnum=DEFAULT_PARTNUM;
	dimnum=DEFAULT_DIM;
	w = DEFAULT_W;
	c1 = DEFAULT_C1;
	c2 = DEFAULT_C2;

	//gets platforms
	platform = getclPlatform();

	//finds gpus
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//gets a context with the first GPU found devices
	context=clCreateContext(NULL , 1, &device, NULL, NULL, &ret);

	//get command queue
	queue=clCreateCommandQueueWithProperties(context, device, NULL, &ret);

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";

		buildSource();

	//if there is a binary, get it and use it
	} else {
		buildBinary(binaryfile);
	}

	//get kernels
	getKernels();

	//set up buffers
	makeBuffers();
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
	platform = getclPlatform();

	//finds gpus
	ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//gets a context with the first GPU found devices
	context=clCreateContext(NULL , 1, &device, NULL, NULL, &ret);

	//get command queue
	queue=clCreateCommandQueueWithProperties(context, device, NULL, &ret);

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";
		buildSource();

	//if there is a binary, get it and use it
	} else {

		buildBinary(binaryfile);
	}

	//get kernels
	getKernels();

	//create buffers for particle positions, velocities, velocities, fitnesses
	makeBuffers();
}

//build program with source
void clSwarm::buildSource(){
	//use C++11 string literals to get kernel
	std::string src{
	#include "kernelstring.cl"
	};

	size_t srcsize = src.size();
	const char * tmpsrc = src.data();
	program = clCreateProgramWithSource(context, 1, (const char **)&tmpsrc, &srcsize, &ret);
	ret = clBuildProgram(program, 1, &device, "  ", NULL, NULL);
	checkBuild(ret,program,device);

	writeBinary(program,"kernels.bin");
}

//build program with binary
void clSwarm::buildBinary(FILE * binaryfile){
	size_t size[1];
	char * tmpbin = readBinary(binaryfile, size);
	program = clCreateProgramWithBinary(context,1,&device, size, (const unsigned char **)&(tmpbin), NULL, &ret);

	delete [] tmpbin;

	ret = clBuildProgram(program, 1, &device, "", NULL, NULL);

	if(ret != CL_SUCCESS){
		std::cerr << "binary build failed, building from source.\n";
		buildSource();
	}
}

void clSwarm::getKernels(){
	distr = clCreateKernel(program, "distribute", &ret);
	cmpre= clCreateKernel(program, "compare",&ret);
	updte=clCreateKernel(program, "update",&ret);
	updte2=clCreateKernel(program, "update2",&ret);
}

void clSwarm::makeBuffers(){
	//create buffers for particle positions, velocities, velocities, fitnesses
	presentbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	vbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	fitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	std::vector<cl_float> tmp(partnum*dimnum,0);

	//set buffers for gbests, and pbests to 0
	gbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//set the vectors values
	for(unsigned int dex = 0; dex <partnum; ++dex)
		tmp[dex]=-HUGE_VALF;

	//create memory buffer for nonparticle fitnesses
	gfitbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(cl_float), NULL,&ret);
	pfitnessbuf=clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float) ,NULL,&ret);
	
	ret= clEnqueueWriteBuffer(queue, gfitbuf, CL_TRUE , 0, sizeof(cl_float),tmp.data(),0,NULL,&outev);
	inev[0]=outev;
	
	ret= clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),0,NULL,&outev);
	inev[1]=outev;
	evnum = 2;

	//make memory pool for upper and lower bounds
	upperboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=clCreateBuffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
}

//the destructor
clSwarm::~clSwarm(){

	//finish and flush everything
	for(cl_mem x : {vbuf, presentbuf, gbestbuf, gfitbuf, pbestbuf, 
					upperboundbuf, lowerboundbuf, pfitnessbuf, fitnessbuf}) {
		ret = clReleaseMemObject(x);
	}

	ret = clFlush(queue);
	ret = clFinish(queue);
	ret = clReleaseCommandQueue(queue);
	for(cl_kernel tmp: {distr,cmpre,updte,updte2}) {
		ret = clReleaseKernel(tmp);
	}

	ret = clReleaseProgram(program);

	ret = clReleaseContext(context);
}

//sets number of particles
void clSwarm::setPartNum(cl_uint num){

	//reset particle swarm #
	partnum=num;

	//needs dimension to initialize
	presentbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	pbestbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	vbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//does not need dimensions
	pfitnessbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	fitnessbuf = clCreateBuffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

	//re-writes values to fitness buffers
	std::vector<cl_float> tmp(partnum,-HUGE_VALF);
	ret = clEnqueueWriteBuffer(queue, gfitbuf,CL_TRUE, 0, partnum*sizeof(cl_float),tmp.data(), 0, NULL,&outev);
	inev[1]=outev;
	ret = clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE, 0, partnum*sizeof(cl_float),tmp.data(), 0, NULL, &outev);
	inev[0]=outev;
	evnum = 2;
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

	ret= clEnqueueWriteBuffer(queue, pfitnessbuf,CL_TRUE, 0, partnum*sizeof(cl_float),tmp.data(), evnum, inev, &outev);
	inev[0]=outev;

	//re-writes values to gfitness
	ret=clEnqueueWriteBuffer(queue, gfitbuf,CL_TRUE, 0, sizeof(cl_float),tmp.data(), 0, NULL,&outev);
	inev[0]=outev;
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
	ret= clEnqueueWriteBuffer(queue, upperboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), upper, 0, NULL, &outev);
	inev[0]=outev;
	
	ret=clEnqueueWriteBuffer(queue, lowerboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), lower, 0, NULL, &outev);
	inev[1]=outev;
	evnum = 2;

	//set kernel args
	ret = clSetKernelArg(distr, 0, sizeof(cl_mem), &lowerboundbuf);
	ret = clSetKernelArg(distr, 1, sizeof(cl_mem), &upperboundbuf);
	ret = clSetKernelArg(distr, 2, sizeof(cl_mem), &presentbuf);
	ret = clSetKernelArg(distr,3, sizeof(cl_uint), &dimnum);
	ret = clSetKernelArg(distr,4, sizeof(cl_uint), &partnum);

	//set up work dimensionts
	std::vector<size_t> dim = {partnum, dimnum};

	ret = clEnqueueNDRangeKernel(queue, distr, 2, NULL, dim.data(), NULL ,0 , NULL, NULL);
}

//run the position and velocity update equation
void clSwarm::update(int times){

	//seed random to use during process
	srand(time(NULL));

	std::vector<size_t> dim = {partnum, dimnum};

	// updates everything
	while(times--){

		//set args for fitness eval
		ret = clSetKernelArg(updte2,0, sizeof(cl_mem), &fitnessbuf);
		ret = clSetKernelArg(updte2,1, sizeof(cl_uint), &dimnum);
		ret = clSetKernelArg(updte2,2, sizeof(cl_mem), &pfitnessbuf);
		ret = clSetKernelArg(updte2,3, sizeof(cl_mem), &presentbuf);
		ret = clSetKernelArg(updte2,4, sizeof(cl_mem), &pbestbuf);
		ret = clSetKernelArg(updte2,5, sizeof(cl_uint),&partnum);

		ret= clEnqueueNDRangeKernel(queue, updte2, 1, NULL, (size_t*)&partnum,NULL, 1, inev, &outev);
		inev[0] = outev;

		//set kernel args
		ret=clSetKernelArg(cmpre,0, sizeof(cl_mem), &presentbuf);
		ret=clSetKernelArg(cmpre,1,sizeof(cl_mem), &gbestbuf);
		ret=clSetKernelArg(cmpre,2, sizeof(cl_mem), &fitnessbuf);
		ret=clSetKernelArg(cmpre,3, sizeof(cl_mem), &gfitbuf);
		ret=clSetKernelArg(cmpre,4, sizeof(cl_uint), &partnum);
		ret=clSetKernelArg(cmpre,5, sizeof(cl_uint), &dimnum);

		//wait then run comparison
		size_t one = 1;
		ret= clEnqueueNDRangeKernel(queue, cmpre, 1,NULL, &one, NULL , 1, inev, &outev);
		inev[0]=outev;

		//set kernel args
		cl_uint seed = rand();
		ret=clSetKernelArg(updte,0, sizeof(cl_mem), &presentbuf);
		ret=clSetKernelArg(updte,1, sizeof(cl_mem), &vbuf);
		ret=clSetKernelArg(updte,2, sizeof(cl_float), &w);
		ret=clSetKernelArg(updte,3, sizeof(cl_uint), &seed);
		ret=clSetKernelArg(updte,4, sizeof(cl_mem), &pfitnessbuf);
		ret=clSetKernelArg(updte,5, sizeof(cl_mem), &upperboundbuf);
		ret=clSetKernelArg(updte,6, sizeof(cl_mem), &pbestbuf);
		ret=clSetKernelArg(updte,7, sizeof(cl_mem), &gbestbuf);
		ret=clSetKernelArg(updte,8, sizeof(cl_mem), &lowerboundbuf);
		ret=clSetKernelArg(updte,9, sizeof(cl_mem), &fitnessbuf);
		ret=clSetKernelArg(updte,10, sizeof(cl_uint), &dimnum);
		ret=clSetKernelArg(updte,11, sizeof(cl_float), &c1);
		ret=clSetKernelArg(updte,12, sizeof(cl_float), &c2);

		//wait then execute
		ret=clEnqueueNDRangeKernel(queue,updte, 2, NULL ,  (const size_t *)dim.data() , NULL, 1, inev, &outev);
		inev[0]=outev;
		evnum = 1;
	}

	//set args for fitness eval
	ret = clSetKernelArg(updte2,0, sizeof(cl_mem), &fitnessbuf);
	ret = clSetKernelArg(updte2,1, sizeof(cl_uint), &dimnum);
	ret = clSetKernelArg(updte2,2, sizeof(cl_mem),  &pfitnessbuf);
	ret = clSetKernelArg(updte2,3, sizeof(cl_mem), &presentbuf);
	ret = clSetKernelArg(updte2,4, sizeof(cl_mem), &pbestbuf);
	ret = clSetKernelArg(updte2,5, sizeof(cl_uint), &partnum);

	ret= clEnqueueNDRangeKernel(queue, updte2, 1, NULL, (const size_t *)&partnum,NULL, 1, inev, &outev);
	inev[0] = outev;

	//set kernel args
	ret=clSetKernelArg(cmpre, 0, sizeof(cl_mem), &presentbuf);
	ret=clSetKernelArg(cmpre, 1, sizeof(cl_mem), &gbestbuf);
	ret=clSetKernelArg(cmpre, 2, sizeof(cl_mem), &fitnessbuf);
	ret=clSetKernelArg(cmpre, 3, sizeof(cl_mem), &gfitbuf);
	ret=clSetKernelArg(cmpre, 4, sizeof(cl_uint), &partnum);
	ret=clSetKernelArg(cmpre, 5, sizeof(cl_uint), &dimnum);

	//wait then run comparison
	size_t one = 1;
	ret= clEnqueueNDRangeKernel(queue, cmpre, 1,NULL, &one, NULL , 1, inev, &outev);
	inev[0]=outev;
	evnum =1;
}

//sets particle data
void clSwarm::setPartData(cl_float * in){
	ret = clEnqueueWriteBuffer(queue, presentbuf,CL_TRUE, 0,partnum*dimnum*sizeof(cl_float),in, 0, NULL, NULL);
}

//copies particle data to the argument
void clSwarm::getPartData(cl_float * out){
	ret = clEnqueueReadBuffer(queue, presentbuf, CL_TRUE, 0, partnum*dimnum*sizeof(cl_float),out, 0, NULL, NULL);
}

//returns the fitness of the best particle
cl_float clSwarm::getGFitness(){

	cl_float out;

	//get value from GPU
	ret=clEnqueueReadBuffer(queue, gfitbuf, CL_TRUE, 0, sizeof(cl_float), &out, 0, NULL ,NULL);

	return out;
}

//returns best position of the clSwarm
void clSwarm::getGBest(cl_float * out){

	//get value from buffer
	ret = clEnqueueReadBuffer(queue, gbestbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), out, 0, NULL, &outev);
	clWaitForEvents(1, &outev);
}

void clSwarm::wait(){
	clWaitForEvents(evnum, inev);
	evnum=0;
}
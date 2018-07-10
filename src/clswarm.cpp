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
		std::string log;
		ret = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.data(), 500, NULL);
		std::cerr << "Build Failed.\nBuild Log:\n" << blog << "\n";
		exit(1);
	}
}

cl::Platform getDefaultPlatform(){
	std::vector<cl::Platform> plats;
	cl::Platform::get(&plats);

	cl::Platform plat;
	for (auto &p : plats) {
		std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if (platver.find("OpenCL 2.") != std::string::npos) {
				plat = p;
		}
	}
	if (plat() == 0)  {
		std::cout << "No OpenCL 2.X platform found.";
		exit(-1);
	}
	return cl::Platform::setDefault(plat);
}

void writeBinary(std::vector<unsigned char> in,const char filename[]){
	FILE * fout = fopen(filename,"wb");
	if(!fout){
		std::cerr << "Could not open file for writing.\n";
		exit(1);
	} 

	fwrite(in.data(),sizeof(char),in.size(), fout);
	fclose(fout);
}

unsigned char * readBinary(FILE * fin, size_t * size){
		
	fseek(fin,0,SEEK_END);
	*size = ftell(fin);
	char * binin = new char[*size];
	rewind(fin);
	fread(binin,sizeof(char),*size,fin);
	
	fclose(fin);

	return binin;
}

//sets dimensions to 1 and number of particles to 100 and w to 1.0
clSwarm::clSwarm(){

	//set swarm characteristics to defaults
	partnum=DEFAULT_PARTNUM;
	dimnum=DEFAULT_DIM;
	w = DEFAULT_W;
	c1 = DEFAULT_C1;
	c2 = DEFAULT_C2;

	//gets platforms
	platform = getDefaultPlatform();

	//finds gpus
	device = cl::Device::getDefault();

	//gets a context with the first GPU found devices
	context=cl::Context(device);

	context=cl::Context(devices);

	//get command queue
	queue=cl::CommandQueue(context,device);

	//use C++11 string literals to get kernel
	const char  src[] =
	R"(#include "kernels.cl")"
	;

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";
		fclose(binaryfile);

		sources.push_back({src,sizeof(src)});

		program=cl::Program(context,sources);

		ret=program.build({device}, " -cl-std=CL2.0 ");

		if(ret!=CL_SUCCESS){
			std::string blog=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			std::cerr << "Build Failed.\nBuild Log:\n" << blog << "\n";
			exit(1);
		}
		writeBinary(program.getInfo<CL_PROGRAM_BINARIES>()[0],"kernels.bin");

	//if there is a binary, get it and use it
	} else {
		std::vector<std::vector<unsigned char>> tmpvec(1, readBinary(binaryfile));
		
		program=cl::Program(context,{device}, tmpvec,NULL, &ret);

		ret = program.build({device}," -cl-std=CL2.0 ");

		if(ret!=CL_SUCCESS){
			std::string blog=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			std::cerr << "Build Failed.\nBuild Log:\n" << blog << "\n";
			exit(1);
		}
	}


	//build kernels
	distr = cl::Kernel(program, "distribute",&ret);
	cmpre= cl::Kernel(program, "compare",&ret);
	updte=cl::Kernel(program, "update",&ret);
	updte2=cl::Kernel(program, "update2",&ret);

	//creates buffers for every resource
	presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));
	pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));
	vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));
	fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	std::vector<cl_float> tmp(partnum*dimnum,0);

	//set buffers for gbests, and pbests to 0
	gbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);

	//set the vectors values
	for(int dex =-1;++dex<partnum;)
		tmp[dex]=-HUGE_VALF;

	//create memory buffer for nonparticle fitnesses
	gfitbuf = cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float),NULL,&ret);
	ret=queue.enqueueWriteBuffer(gfitbuf, CL_TRUE , 0, sizeof(cl_float),tmp.data(),NULL,&ev);
	evs.emplace_back(ev);
	pfitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	ret=queue.enqueueWriteBuffer(pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),NULL,&ev);
	evs.emplace_back(ev);

	//make memory pool for upper and lower bounds
	upperboundbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float),NULL,&ret);
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
	const char  src[] =
	R"(#include "kernels.cl")"
	;

	FILE * binaryfile = fopen("kernels.bin","rb");

	//if no binary exists, get source and write out binary
	if(binaryfile == NULL){
		std::cout << "No binary file for kernel found, Compiling.\n";
		fclose(binaryfile);

		size_t srcsize = sizeof(src);
		program = clCreateProgramWithSource(context, 1, &src, &srcsize, &ret);
		ret = clBuildProgram(program, 1, &device, " -cl-std=CL2.1 ", NULL, NULL);
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

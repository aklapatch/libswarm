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

//sets dimensions to 1 and number of particles to 100 and w to 1.0
clSwarm::clSwarm(){

	//set swarm characteristics to defaults
	partnum=DEFAULT_PARTNUM;
	dimnum=DEFAULT_DIM;
	w = DEFAULT_W;
	c1 = DEFAULT_C1;
	c2 = DEFAULT_C2;

	//gets platforms
	ret =cl::Platform::get(&platforms);

	//finds devices
	ret= platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

	//gets a context with the first GPU found devices
	context=cl::Context(devices);

	//get command queue
	queue=cl::CommandQueue(context,devices[0]);

	//use C++11 string literals to get kernel
	const char  src[] =
	#include "kernelstring.cl"
	;


	FILE * binaryfile = fopen("kernels.bin","rb");

	char * binin;
	long int size=0;
	if(binaryfile == NULL){
		std::cerr << "No binary file for kernel found, Compiling.\n";

		auto binary = program.getInfo<CL_PROGRAM_BINARIES>(); 

		//store the kernel in the sources object
		sources.push_back({src,sizeof(src)});

		//init and build program
		program=cl::Program(context,sources);

		FILE * binout = fopen("kernels.bin","wb");
		if(!binout){
			std::cerr << "Could not open file for writing.\n";
			exit(1);
		} 

		fwrite(binary[0].data(),sizeof(char),binary[0].size(),binout);
		fclose(binout);

	} else {
		fseek(binaryfile,0,SEEK_END);
		size = ftell(binaryfile);
		binin = new char[size];
		rewind(binaryfile);
		fread(binin,sizeof(char),size,binaryfile);
		std::vector<std::vector<unsigned char>> binaries;
		fclose(binaryfile);
		binaries.emplace_back(std::vector<unsigned char>(binin,binin + size));
		delete [] binin;
		program=cl::Program(context,devices, binaries,NULL, &ret);
	}

	//build program
	ret=program.build(devices, " -cl-std=CL2.0 ");

	if(ret!=CL_SUCCESS){
		std::string blog=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		std::cerr << "Build Failed.\n Build Log:\n" << blog << "\n";
		exit(1);
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
	ret=cl::Platform::get(&platforms);

	//finds gpus
	ret=platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	//gets a context with the first GPU found devices
	context=cl::Context(devices);

	//get command queue
	queue=cl::CommandQueue(context,devices[0]);

	//use C++11 string literals to get kernel
	const char  src[] =
	#include "kernelstring.cl"
	;

	FILE * binaryfile = fopen("kernels.bin","rb");

	char * binin;
	long int size=0;
	if(binaryfile == NULL){
		std::cerr << "No binary file for kernel found, Compiling.\n";

		auto binary = program.getInfo<CL_PROGRAM_BINARIES>(); 

		//store the kernel in the sources object
		sources.push_back({src,sizeof(src)});

		//init and build program
		program=cl::Program(context,sources);

		FILE * binout = fopen("kernels.bin","wb");
		if(!binout){
			std::cerr << "Could not open file for writing.\n";
			exit(1);
		} 

		fwrite(binary[0].data(),sizeof(char),binary[0].size(),binout);
		fclose(binout);

	} else {
		fseek(binaryfile,0,SEEK_END);
		size = ftell(binaryfile);
		binin = new char[size];
		rewind(binaryfile);
		fread(binin,sizeof(char),size,binaryfile);
		std::vector<std::vector<unsigned char>> binaries;
		fclose(binaryfile);
		binaries.emplace_back(std::vector<unsigned char>(binin,binin + size));
		delete [] binin;
		program=cl::Program(context,devices, binaries,NULL, &ret);
	}

	//build program
	ret=program.build(devices, " -cl-std=CL2.0 ");

	if(ret!=CL_SUCCESS){
		std::string blog=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
		std::cerr << "Build Failed.\nBuild Log:\n" << blog << "\n";
		exit(1);
	}

	//get kernels
	distr = cl::Kernel(program, "distribute",&ret);
	cmpre= cl::Kernel(program, "compare",&ret);
	updte=cl::Kernel(program, "update",&ret);
	updte2=cl::Kernel(program, "update2",&ret);

	//create buffers for particle positions, velocities, velocities, fitnesses
	presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);

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
	upperboundbuf=cl::Buffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=cl::Buffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
}

//the destructor
clSwarm::~clSwarm(){

	//finish and flush out queue
	queue.flush();
	queue.finish();
}

//sets number of particles
void clSwarm::setPartNum(cl_uint num){

	//reset particle swarm #
	partnum=num;

	std::vector<cl_float> tmp(partnum*dimnum,0);

	//needs dimension to initialize
	presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float),NULL,&ret);
	vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	for(int i = -1; ++i<partnum;)
		tmp[i]=-HUGE_VALF;

	//does not need dimensions
	pfitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float),NULL,&ret);
	ret=queue.enqueueWriteBuffer(pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),&evs,&ev);
	evs.emplace_back(ev);

	fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));

	//re-writes values to gfitness
	ret=queue.enqueueWriteBuffer(gfitbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),NULL,&ev);
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
	gbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));

	presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

	//make memory pool for upper and lower bounds
	upperboundbuf=cl::Buffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);
	lowerboundbuf=cl::Buffer(context, CL_MEM_READ_ONLY,dimnum*sizeof(cl_float),NULL,&ret);

	std::vector<cl_float> tmp(partnum,-HUGE_VALF);

	ret=queue.enqueueWriteBuffer(pfitnessbuf,CL_TRUE,0, partnum*sizeof(cl_float),tmp.data(),&evs,&ev);
	evs.emplace_back(ev);

	//re-writes values to gfitness
	ret=queue.enqueueWriteBuffer(gfitbuf,CL_TRUE,0, sizeof(cl_float),tmp.data(),NULL,&ev);
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
	ret=queue.enqueueWriteBuffer(upperboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), upper,&evs,&ev);
	evs.emplace_back(ev);
	ret=queue.enqueueWriteBuffer(lowerboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), lower,NULL,&ev);
	evs.emplace_back(ev);

	//set kernel args
	ret=distr.setArg(0,lowerboundbuf);
	ret=distr.setArg(1,upperboundbuf);
	ret=distr.setArg(2,presentbuf);
	ret=distr.setArg(3,dimnum);
	ret=distr.setArg(4,partnum);

	//execute
	ret=queue.enqueueNDRangeKernel(distr,cl::NullRange, cl::NDRange(partnum,dimnum),cl::NullRange,&evs,&ev);
	evs.emplace_back(ev);
}

size_t rng(){
    static size_t x = 123456789;
    static size_t y = 362436069;
    static size_t z = 521288629;
    static size_t w = 88675123;
    size_t t;
    t = x ^ (x << 11);   
    x = y; 
    y = z; 
    z = w;   
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

//run the position and velocity update equation
void clSwarm::update(unsigned int times){

	//set up memory to take the random array
	unsigned int size=2*partnum*dimnum;
	cl_float * ran = new cl_float [size];
	cl::Buffer ranbuf(context, CL_MEM_READ_ONLY,size*sizeof(cl_float));
	unsigned int i;

	// updates everything
	while(times--){

		//set args for fitness eval
		ret=updte2.setArg(0,fitnessbuf);
		ret=updte2.setArg(1,dimnum);
		ret=updte2.setArg(2,pfitnessbuf);
		ret=updte2.setArg(3,presentbuf);
		ret=updte2.setArg(4,pbestbuf);
		ret=updte2.setArg(5,partnum);

		ret=queue.enqueueNDRangeKernel(updte2,cl::NullRange,cl::NDRange(partnum),cl::NullRange,&evs, &ev);
		evs.emplace_back(ev);

		//set kernel args
		ret=cmpre.setArg(0,presentbuf);
		ret=cmpre.setArg(1,gbestbuf);
		ret=cmpre.setArg(2,fitnessbuf);
		ret=cmpre.setArg(3,gfitbuf);
		ret=cmpre.setArg(4,partnum);
		ret=cmpre.setArg(5,dimnum);

		//wait then run comparison
		ret=queue.enqueueNDRangeKernel(cmpre,cl::NullRange,cl::NDRange(1),cl::NullRange,&evs, &ev);
		evs.emplace_back(ev);

		//make a array of random numbers
		for(i=0; ++i < size;)
			ran[i] = RAN;

		//write random numbers to buffer
		queue.enqueueWriteBuffer(ranbuf, CL_TRUE, 0, size*sizeof(cl_float), ran,&evs,&ev);
		evs.emplace_back(ev);

		//set kernel args
		ret=updte.setArg(0,presentbuf);
		ret=updte.setArg(1,vbuf);
		ret=updte.setArg(2,w);
		ret=updte.setArg(3,ranbuf);
		ret=updte.setArg(4,pfitnessbuf);
		ret=updte.setArg(5,upperboundbuf);
		ret=updte.setArg(6,pbestbuf);
		ret=updte.setArg(7,gbestbuf);
		ret=updte.setArg(8,lowerboundbuf);
		ret=updte.setArg(9,fitnessbuf);
		ret=updte.setArg(10,dimnum);
		ret=updte.setArg(11,c1);
		ret=updte.setArg(12,c2);

		//wait then execute
		ret=queue.enqueueNDRangeKernel(updte,cl::NullRange, cl::NDRange(partnum,dimnum) , cl::NullRange, &evs, &ev);
		evs.emplace_back(ev);
	}
	delete [] ran;

	//evaluate fitness one more time
	//set args for fitness eval
	ret=updte2.setArg(0,fitnessbuf);
	ret=updte2.setArg(1,dimnum);
	ret=updte2.setArg(2,pfitnessbuf);
	ret=updte2.setArg(3,presentbuf);
	ret=updte2.setArg(4,pbestbuf);
	ret=updte2.setArg(5,partnum);

	//wait then execute kernel
	ret=queue.enqueueNDRangeKernel(updte2,cl::NullRange, cl::NDRange(partnum) ,cl::NullRange, &evs, &ev);
	evs.emplace_back(ev);

	//set kernel args
	ret=cmpre.setArg(0,presentbuf);
	ret=cmpre.setArg(1,gbestbuf);
	ret=cmpre.setArg(2,fitnessbuf);
	ret=cmpre.setArg(3,gfitbuf);
	ret=cmpre.setArg(4,partnum);
	ret=cmpre.setArg(5,dimnum);

	//compare one more time
	ret=queue.enqueueNDRangeKernel(cmpre,cl::NullRange,cl::NullRange,cl::NullRange,&evs,&ev);
	evs.emplace_back(ev);
}

//sets particle data
void clSwarm::setPartData(cl_float * in){
	queue.enqueueWriteBuffer(presentbuf,CL_TRUE, 0,partnum*dimnum*sizeof(cl_float),in,&evs,&ev);
	evs.emplace_back(ev);
}

//copies particle data to the argument
void clSwarm::getPartData(cl_float * out){
	queue.enqueueReadBuffer(presentbuf,CL_TRUE,0,partnum*dimnum*sizeof(cl_float),out,&evs,&ev);
	evs.emplace_back(ev);
}

//returns the fitness of the best particle
cl_float clSwarm::getGFitness(){

	cl_float out;

	//get value from GPU
	ret=queue.enqueueReadBuffer(gfitbuf, CL_TRUE, 0,sizeof(cl_float), &out,&evs,&ev);
	evs.emplace_back(ev);

	return out;
}

//returns best position of the clSwarm
void clSwarm::getGBest(cl_float * out){

	//get value from buffer
	ret=queue.enqueueReadBuffer(gbestbuf, CL_TRUE, 0,dimnum*sizeof(cl_float),out,&evs, &ev);
	evs.emplace_back(ev);
}

void clSwarm::wait(){
	queue.enqueueBarrierWithWaitList(&evs);
}
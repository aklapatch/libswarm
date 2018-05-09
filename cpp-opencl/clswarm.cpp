/*clswarm.cpp
implementation for swarm
for opencl version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "clswarm.hpp"
#include <iostream>

///sets particle data
void clswarm::setPartData(cl_float * in){
	queue.enqueueWriteBuffer(presentbuf,CL_TRUE, 0,partnum*dimnum*sizeof(cl_float),in);
}

///returns particle data
cl_float * clswarm::getPartData(){

	///store particle data
    cl_float *out=new cl_float [partnum*dimnum];
    queue.enqueueReadBuffer(presentbuf, CL_TRUE, 0,partnum*dimnum*sizeof(cl_float),out);

    return out;
}

///returns particle number
cl_uint clswarm::getPartNum(){
    return partnum;
}

///returns dimension number
cl_uint clswarm::getDimNum(){
    return dimnum;
}

///sets dimensions to 1 and number of particles to 100 and w to 1.0
clswarm::clswarm(){

    ///set swarm characteristics to defaults
    partnum=DEFAULT_PARTNUM;
    dimnum=DEFAULT_DIM;
    w = DEFAULT_W;
    c1=C1;
    c2=C2;
    gfitness=-HUGE_VAL;    

    ///gets platforms
	cl::Platform::get(&platforms);
	
	///finds gpus
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	
	///gets a context with the first GPU found devices
    context=cl::Context({devices[0]});

	///get command queue
    queue=cl::CommandQueue(context,devices[0]);

    ///use C++11 string literals to get kernel
    const char  src[] =
    #include "kernelstring.cl"
    ;

    ///store the kernel in the sources object
    sources.push_back({src,sizeof(src)});

    ///init and build program
    program=cl::Program(context,sources);
    ret=program.build({devices[0]});    
    
    ///build program
    program.build({devices[0]});
	
    ///build kernels
	distr = cl::Kernel(program, "distribute");
    cmpre= cl::Kernel(program, "compare");
    updte=cl::Kernel(program, "update");
    updte2=cl::Kernel(program, "update2");

    ///make buffers for particle and dimension numbers
    dimnumbuf=cl::Buffer(context, CL_MEM_READ_ONLY,sizeof(cl_int));
    ret=queue.enqueueWriteBuffer(dimnumbuf, CL_TRUE, 0, sizeof(cl_int), &dimnum);
    partnumbuf=cl::Buffer(context, CL_MEM_READ_ONLY,sizeof(cl_int));
    ret=queue.enqueueWriteBuffer(partnumbuf, CL_TRUE, 0, sizeof(cl_int), &partnum);

    ///create memory buffer for gfit and write to it
    gfitbuf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(gfitbuf, CL_TRUE, 0, sizeof(cl_float), &gfitness);

    ///create and write w to memory
    wbuf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(wbuf, CL_TRUE, 0, sizeof(cl_float), &w);

    ///get memory for and write to c1buffer
    c1buf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(c1buf, CL_TRUE, 0, sizeof(cl_float), &c1);

    ///get memory for and write to c2buffer
    c2buf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(c2buf, CL_TRUE, 0, sizeof(cl_float), &c2);

    ///creates buffers for every resource
    presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));
    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));
    vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));
    fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    ///set buffers for pfitnesses, gbests, and pbests
    pfitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));
    gbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));
    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    ///set args for initialization kernel
    initpfit.setArg(0,pfitnessbuf);
    ret=queue.enqueueNDRangeKernel(initpfit,cl::NullRange, cl::NDRange(partnum),cl::NullRange);   

    ///init bests as 0
    initzero.setArg(0,gbestbuf);
    ret=queue.enqueueNDRangeKernel(initzero,cl::NullRange, cl::NDRange(dimnum),cl::NullRange);
    initzero.setArg(0,pbestbuf);    
    ret=queue.enqueueNDRangeKernel(initzero,cl::NullRange, cl::NDRange(partnum*dimnum),cl::NullRange);
}

///sets dimensions to 1 and number of particles to 100 and w to 1.5
//! TODO, convert all unsigned ints to cl_uint type
clswarm::clswarm(cl_uint numdims, cl_uint numparts,cl_float inw, cl_float c1in, cl_float c2in){

    ///set swarm characteristics
    partnum=numparts;
    dimnum=numdims;
    w = inw;
    gfitness=-HUGE_VALF;    
    c1=c1in;
    c2=c2in;

   ///gets platforms
	ret=cl::Platform::get(&platforms);
	
	///finds gpus
	ret=platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	
	///gets a context with the first GPU found devices
    context=cl::Context({devices[0]});

	///get command queue
    queue=cl::CommandQueue(context,devices[0]);

    ///use C++11 string literals to get kernel
    const char  src[] =
    #include "kernelstring.cl"
    ;

    ///store the kernel in the sources object
    sources.push_back({src,sizeof(src)});    

    ///init and build program
    program=cl::Program(context,sources);
    ret=program.build({devices[0]});

    std::string blog=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
    std::cerr << "Build Log:\n" << blog << "\n";

    ///get kernels
	distr = cl::Kernel(program, "distribute");
    cmpre= cl::Kernel(program, "compare");
    updte=cl::Kernel(program, "update");
    updte2=cl::Kernel(program, "update2");

    ///initialization kernels
    initpfit=cl::Kernel(program, "initpfit");
    initzero=cl::Kernel(program, "initzero");

    ///make buffers for particle and dimension numbers
    dimnumbuf=cl::Buffer(context, CL_MEM_READ_ONLY,sizeof(cl_uint));
    ret=queue.enqueueWriteBuffer(dimnumbuf, CL_TRUE, 0, sizeof(cl_uint), &dimnum);
    partnumbuf=cl::Buffer(context, CL_MEM_READ_ONLY,sizeof(cl_uint));
    ret=queue.enqueueWriteBuffer(partnumbuf, CL_TRUE, 0, sizeof(cl_uint), &partnum);

    ///create memory buffer for gfit and write to it
    gfitbuf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(gfitbuf, CL_TRUE, 0, sizeof(cl_float), &gfitness);

    ///create and write w to memory
    wbuf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(wbuf, CL_TRUE, 0, sizeof(cl_float), &w);

    ///get memory for and write to c1buffer
    c1buf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(c1buf, CL_TRUE, 0, sizeof(cl_float), &c1);

    ///get memory for and write to c2buffer
    c2buf=cl::Buffer(context, CL_MEM_READ_WRITE,sizeof(cl_float));
    ret=queue.enqueueWriteBuffer(c2buf, CL_TRUE, 0, sizeof(cl_float), &c2);

    ///creates buffers for every resource
    presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));

    vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    ///set buffers for pfitnesses, gbests, and pbests
    pfitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));
    gbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));
    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    ///set args for initialization kernel
    initpfit.setArg(0,pfitnessbuf);
    ret=queue.enqueueNDRangeKernel(initpfit,cl::NullRange, cl::NDRange(partnum),cl::NullRange);   

    ///init bests as 0
    initzero.setArg(0,gbestbuf);
    ret=queue.enqueueNDRangeKernel(initzero,cl::NullRange, cl::NDRange(dimnum),cl::NullRange);
    initzero.setArg(0,pbestbuf);    
    ret=queue.enqueueNDRangeKernel(initzero,cl::NullRange, cl::NDRange(partnum*dimnum),cl::NullRange);
}

///the destructor
clswarm::~clswarm(){

    //finish and flush out queue
    queue.flush();
    queue.finish();
}

///sets number of particles
void clswarm::setPartNum(cl_uint num){

    ///reset particle swarm #
    partnum=num;

    ///does not need dimensions
    pfitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));

    fitnessbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*sizeof(cl_float));

    ///needs dimension to initialize
    presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    //!TODO make kernel to reinit values for these buffers
}

///sets number of dimensions
void clswarm::setDimNum(cl_uint num){

    ///set dimension number    
    dimnum=num;

    ///recreates buffers for every resource
    //! TODO kernel value init here
    gbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));

    presentbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    pbestbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));

    vbuf=cl::Buffer(context, CL_MEM_READ_WRITE,partnum*dimnum*sizeof(cl_float));
}

///set inertial weight
void clswarm::setWeight(cl_float nw){

	///store the inertial weight
	w=nw;

    ///write w to memory
    queue.enqueueWriteBuffer(wbuf, CL_TRUE, 0, sizeof(cl_float), &nw);
}

///return inertial weight
cl_float clswarm::getWeight(){
	return w;
}

///set behavioral constants
void clswarm::setConstants(cl_float nc1,cl_float nc2){
    
	///store constants
	c1=nc1;
	c2=nc2;
	
	///write to new constants to buffers
    queue.enqueueWriteBuffer(c1buf, CL_TRUE, 0, sizeof(cl_float), &nc1);
    queue.enqueueWriteBuffer(c2buf, CL_TRUE, 0, sizeof(cl_float), &nc2);
}

//return constants
cl_float * clswarm::getConstants(){
	cl_float * out = new cl_float[2];
	out[0]=c1;
	out[1]=c2;
	return out;
}

///distribute particle linearly from lower bound to upper bound
void clswarm::distribute(cl_float * lower, cl_float * upper){
    
    ///make memory pool
    upperboundbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));
    lowerboundbuf=cl::Buffer(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));

    ///store bounds for later
    ret=queue.enqueueWriteBuffer(upperboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), upper);
    ret=queue.enqueueWriteBuffer(lowerboundbuf, CL_TRUE, 0, dimnum*sizeof(cl_float), lower);

    ///allocate memory for delta buffer
    cl::Buffer deltabuf(context, CL_MEM_READ_WRITE,dimnum*sizeof(cl_float));

    ///set kernel args
	ret=distr.setArg(0,lowerboundbuf);
    ret=distr.setArg(1,upperboundbuf);
    ret=distr.setArg(2,deltabuf);
    ret=distr.setArg(3,presentbuf);
    ret=distr.setArg(4,pbestbuf);
    ret=distr.setArg(5,dimnumbuf);
    ret=distr.setArg(6,partnumbuf);

    ///execute
	ret=queue.enqueueNDRangeKernel(distr,cl::NullRange, cl::NDRange(partnum,dimnum),cl::NullRange);
}

///run the position and velocity update equation
void clswarm::update(unsigned int times){

    ///make random number generator C++11
    std::random_device gen;
    std::uniform_real_distribution<float> distr(1,0);

    ///set up memory to take the random array
    cl_float * ran = new cl_float [(1+dimnum)*partnum];
    cl::Buffer ranbuf(context, CL_MEM_READ_WRITE,(dimnum+1)*partnum*sizeof(cl_float));
    unsigned int i;

    while(times--){

        ///make a array of random numbers
        for(i=0;i<(dimnum+1)*partnum;++i){
            ran[i]= distr(gen);
        }

        ///write random numbers to buffer
        queue.enqueueWriteBuffer(ranbuf, CL_TRUE, 0, (dimnum+1)*partnum*sizeof(cl_float), ran);

        ///set kernel args
	    ret=updte.setArg(0,presentbuf);
        ret=updte.setArg(1,vbuf);
        ret=updte.setArg(2,wbuf);
        ret=updte.setArg(3,ranbuf);
        ret=updte.setArg(4,pfitnessbuf);
        ret=updte.setArg(5,upperboundbuf);
        ret=updte.setArg(6,pbestbuf);
        ret=updte.setArg(7,gbestbuf);
        ret=updte.setArg(8,lowerboundbuf);
        ret=updte.setArg(9,fitnessbuf);
        ret=updte.setArg(10,dimnumbuf);
        ret=updte.setArg(11,c1buf);
        ret=updte.setArg(12,c2buf);

        ///execute kernel
        ret=queue.enqueueNDRangeKernel(updte,cl::NullRange, cl::NDRange(partnum,dimnum),cl::NullRange);

        ///set args for compare kernel
        ret=updte2.setArg(0,fitnessbuf);
        ret=updte2.setArg(1,dimnumbuf);
        ret=updte2.setArg(2,pfitnessbuf);
        ret=updte2.setArg(3,presentbuf);
        ret=updte2.setArg(4,pbestbuf);
        ret=updte2.setArg(5,partnumbuf);

        ret=queue.enqueueNDRangeKernel(updte2,cl::NullRange, cl::NDRange(partnum),cl::NullRange);

        ///set kernel args
	    ret=cmpre.setArg(0,presentbuf);
        ret=cmpre.setArg(1,gbestbuf);
        ret=cmpre.setArg(2,fitnessbuf);
        ret=cmpre.setArg(3,gfitbuf);
        ret=cmpre.setArg(4,partnumbuf);
        ret=cmpre.setArg(5,dimnumbuf);

        ret=queue.enqueueTask(cmpre,NULL,NULL);
    }

    delete [] ran;
}

///returns best position of the clswarm
cl_float * clswarm::getGBest(){
    cl_float * gbest = new cl_float[dimnum];

    ///get value from buffer
    ret=queue.enqueueReadBuffer(gbestbuf, CL_TRUE, 0,dimnum*sizeof(cl_float),gbest);
	
    return gbest;
}   

///returns the fitness of the best particle
cl_float clswarm::getGFitness(){

    ///get value from GPU
    ret=queue.enqueueReadBuffer(gfitbuf, CL_TRUE, 0,sizeof(cl_float), &gfitness);

    return gfitness;
}

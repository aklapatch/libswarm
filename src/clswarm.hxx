/*clswarm.hpp
houses class and function prototypes for swarm class
for opencl version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#ifndef _CLSWARM_HXX_
#define _CLSWARM_HXX_


//set opencl version
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

#ifdef __APPLE__	
#include <OpenCL/opencl.hpp>	
#else	
#include <CL/cl2.hpp>	
#endif	

//macro to check for opencl Errors
#define CHK if(ret!= CL_SUCCESS) { std::cerr << "Error code " << ret << " at Line " << __LINE__ << "\n"; }

#define KER_SIZE 0x100000
#define PLATFORM_NUM 1
#define DEVICE_NUM 1
#define DEFAULT_DIM 1
#define DEFAULT_PARTNUM 100
#define DEFAULT_W 1
#define DEFAULT_C1 1.492
#define DEFAULT_C2 2

template <typename T>
void printbuf(cl::Buffer buf, size_t size,cl::CommandQueue q){
	T * out = new T[size];
	
	int ret=q.enqueueReadBuffer(buf,CL_TRUE,0,size*sizeof(T),out);
	
	if(ret!=CL_SUCCESS)
		std::cerr << " Read error, code: " << ret << "\n";
	
	for (int i=-1;++i<size;)
		std::cout << "Buffer contents[" << i << "] = " << out[i] << "\n";
		
	delete [] out;
	
}

class clswarm {
	private:
	
	// used for debugging and error checking
	cl_int ret;

        // Number of particles and Number of dimensions for each particle
        cl_uint partnum, dimnum;
        
        //buffers for the particle number and dimension numbers
        cl::Buffer dimnumbuf, partnumbuf;

        //best particle dimensions, its fitness, swarm bounds
        cl::Buffer gbestbuf, upperboundbuf, lowerboundbuf;

        //set that so all fitness numbers will show up
        cl::Buffer gfitbuf;

        //inertial weight and 2 behavioral constants
        cl::Buffer wbuf, c1buf, c2buf;

        //particle data
        cl::Buffer presentbuf, pbestbuf, vbuf;
        cl::Buffer pfitnessbuf, fitnessbuf;

        //opencl items
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        cl::Context context;
        cl::Program::Sources sources;
        cl::Program program;
        cl::CommandQueue queue;
        cl::Kernel distr, cmpre, updte, updte2;
        std::vector<cl::Event> evs;
        cl::Event ev;

    public:
        /// Default constructor.
        /// Makes a swarm with 100 particles and a 1 dimension problem space.
        clswarm();

        /** Makes a Swarm with the specified parameters.
         * @param numparts Number of particles in the swarm.
         * @param numdims Number of dimensions in problem space.
         * @param w How much velocity carries over from update to update. Best left between 0 and 1.
         * @param c1 How much the particle's best position affects the particle's velocity.
         * @param c2 How much the global best affects the individual particle's velocity.
         */
        clswarm(cl_uint numparts, cl_uint numdims,cl_float w,cl_float c1,cl_float c2);

        /// Destructor.
        /// Frees Heap-allocated memory
        ~clswarm();

        /// Sets the number of particles in the swarm.
        /// Deletes all stored particle data
        void setPartNum(cl_uint);

        /// Returns number of particles in the swarm.
        cl_uint getPartNum();

        /// Sets number of dimensions in the problem space.
        /// Deletes all particle data
        void setDimNum(cl_uint);

        /// Returns number of dimensions in problem space
	cl_uint getDimNum();

     	/// Sets the inertial weight.
        void setWeight(cl_float);

	/// Returns the inertial weight.
	cl_float getWeight();

	/// Sets the constant affecting the pull of particles' best.
	void setC1(cl_float);

        /// Returns behavioral constant that determines the pull of the particle's best.
	cl_float getC1();
		
        ///  Sets the constant affecting the pull of the global best.
	void setC2(cl_float);
		
	/// Returns global best behavioral constant.
	cl_float getC2();

        /** Sets upper and lower bounds for the problem space and distributes linearly between them.
         * @param lowerbound The lower bound in the problemspace.
         * @param upperbound The upper bound in the problem space.
         */ 
        void distribute(cl_float * upperbound , cl_float * lowerbound);

        /** Updates all particle data values
         * @param times The number of times to update the particle values.
         */
        void update(unsigned int times);

	/// Copies data from the input argument into the particle array.
	void setPartData(cl_float *);

        /// Copies particle data into input argument.
        void getPartData(cl_float *);

        ///returns the fitness of the best particle
        cl_float getGFitness();
		
	/// Copies particle data into the input argument.
        void getGBest(cl_float *);
        
        /// Waits for all pending OpenCL events to compete
        void wait();
};
#endif  //CLSWARM_HXX

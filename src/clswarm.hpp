/*clswarm.hpp
houses class and function prototypes for swarm class
for opencl version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#ifndef _CLSWARM_HPP_
#define _CLSWARM_HPP_

//set opencl version
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define RAN (cl_float)(rng()%1000)/1000

#include <bits/stdc++.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

//macro to check for opencl Errors
#define CHK if(ret!= CL_SUCCESS) { std::cerr << "Error code " << ret << " at Line " << __LINE__ << "\n"; }

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

class clSwarm {
	private:
		// used for debugging and error checking
		cl_int ret;

		// Number of particles and Number of dimensions for each particle
		cl_uint partnum, dimnum;

		//best particle dimensions, its fitness, swarm bounds
		cl_mem gbestbuf, upperboundbuf, lowerboundbuf;

		//set that so all fitness numbers will show up
		cl_mem gfitbuf;

		//inertial weight and 2 behavioral constants
		cl_float w, c1, c2;

		//particle data
		cl_mem presentbuf, pbestbuf, vbuf;
		cl_mem pfitnessbuf, fitnessbuf;

		//opencl items
		std::vector<cl_event> evs;
		cl_platform_id platform;
		cl_device_id device;
		cl_context context;
		cl_program program;
		cl_command_queue queue;
		cl_kernel distr, cmpre, updte, updte2;
		cl_event ev;

	public:
		/// Default constructor.
		/// Makes a swarm with 100 particles and a 1 dimension problem space.
		clSwarm();

		/** Makes a Swarm with the specified parameters.
		 * @param numparts Number of particles in the swarm.
		 * @param numdims Number of dimensions in problem space.
		 * @param w How much velocity carries over from update to update. Best left between 0 and 1.
		 * @param c1 How much the particle's best position affects the particle's velocity.
		 * @param c2 How much the global best affects the individual particle's velocity.
		 */
		clSwarm(cl_uint numparts, cl_uint numdims,cl_float w,cl_float c1,cl_float c2);

		/// Destructor.
		/// Finishes and flushes command Queue
		~clSwarm();

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
#endif  //CLSWARM_HPP

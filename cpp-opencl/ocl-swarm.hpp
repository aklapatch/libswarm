/*particle.hpp
houses class and function prototypes for swarm class
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#ifndef _OCL-SWARM_HPP_
#define _OCL-SWARM_HPP_

#include <vector>
#include <cmath>
#include <random>

#ifdef __APPLE__	
#include <OpenCL/opencl.h>	
#else	
#include <CL/cl.h>	
#endif	

#define KER_SIZE 0x100000
#define PLATFORM_NUM 3
#define DEVICE_NUM 3
#define DEFAULT_DIM 1
#define DEFAULT_PARTNUM 100
#define DEFAULT_W 1
#define C1 1.492
#define C2 2

///there is no particle class, the swarm class has all the data
///that might change later on
class swarm {
    private:
        /// no. of particles, no. of dimensions
        cl_uint dimnum=DEFAULT_DIM,partnum=DEFAULT_PARTNUM;
        cl_mem dimnumbuf, partnumbuf;

        ///best particle dimensions, its fitness, swarm bounds
        cl_mem gbestbuf, upperboundbuf, lowerboundbuf;

        ///set that so all fitness numbers will show up
        cl_float gfitness=-HUGE_VAL;
        cl_mem gfitbuf;

        ///inertial weight and 2 behavioral constants
        cl_float w=DEFAULT_W, c1=C1, c2=C2;
        cl_mem wbuf, c1buf, c2buf;

        ///particle data
        cl_mem presentbuf, pbestbuf, vbuf;
        cl_mem pfitnessbuf, fitnessbuf;
		
		///all the opencl stuff
		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;
		cl_context context = NULL;
		cl_command_queue command_queue = NULL;
		cl_program program = NULL;
		cl_kernel distr = NULL, updte=NULL,cmpre=NULL;
		cl_uint ret_num_devices;
		cl_uint ret_num_platforms;
		cl_int ret;

    public:
        ///defaults to 100 particles and 1 dimension
        swarm();

        ///sets no. particles and no. dimensions and w
        swarm(unsigned int, unsigned int,cl_float);

        ///frees all swarm memory
        ~swarm();

        ///sets the number of particles
        void setpartnum(unsigned int);

        ///sets no. of dimensions
        void setdimnum(unsigned int);

        ///sets inertial weight
        void setweight(cl_float);

        ///sets 2 behavioral constants of the swarm
        void setconstants(cl_float, cl_float);

        /// sets upper and lower bounds and distributes linearly between them
        /** lower bound is first argument, upper bound is second argument */
        void distribute(cl_float* , cl_float* );

        /// updates (int) number of times with *fitness as a fitness function
        void update(cl_uint, cl_float (*fitness)(float*) );

        ///returns the best position in the swarm.
        cl_float* getgbest();

        ///returns the fitness of the best particle
        cl_float getgfitness();
		
		///returns ret for debugging purposes
		cl_int getRet();
};

#endif
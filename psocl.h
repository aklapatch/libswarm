#ifndef PSOCL_H
#define PSOCL_H
/*psocl.h
houses structs and function prototypes
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define PRINTLN printf("line %d\n",__LINE__)

//struct for each particle 
typedef struct clparticle {
    cl_float clfitness, clpfitness;
    cl_mem clpresent, clpbest, clv;
} particle;

typedef struct clenv {
    cl_platform_id=NULL;
    cl_device_id device_id=NULL;
    cl_context context=NULL;
    cl_uint ret_num_devices;
    cl_command_queue command_queue=NULL;
    cl_int ret;
    cl_program program;
    clswarm* data;
}

//struct for the opencl swarm
typedef struct clswarm {
    clparticle *clschool;
    cl_int clpartnum, cldimnum;
    cl_float clgfitness, clw;
    cl_mem clgbest, clbounds;
} clswarm;

clenv* clinit(int dimensionnum, int partnum, float w);

//swarm particle distribution
void cldistributeparticles(clswarm * school,float * bounds);

//running the swarm
void clrunswarm(int iterations, clswarm *school,float (*fitness)(float *));

//Run the swarm conditionally. The keep_going function should return 0 to keep going and 1 to stop
//inputting 0 into the iterations argument will make the swarm run until the function tells it to stop.
//the float * being passed into keep_going is a array of 5
void clconditionalrunswarm(int iterations, clswarm *school,float (*fitness)(float *), int (*keep_going)(float *));

//get best solution
//returns school.gbest
float * clreturnbest(clswarm * school);

//releasing/ending the swarm
void clreleaseswarm(clswarm * school);

#endif
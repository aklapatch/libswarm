#ifndef PSOCL_H
#define PSOCL_H
/*psocl.h
houses structs and function prototypes

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
typedef struct particle {
    double *present, *pbest, fitness, pfitness,*v;
} particle;

//struct for the swarm
typedef struct swarm {
    int partnum, dimnum;
    double *gbest, gfitness,*bounds;
    float w;
    particle *school;
} swarm;

//struct for the opencl swarm
typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, gfitness,*bounds;
    float w;
    particle *school;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_int ret;
} clswarm;

//swarm initializaion
swarm * initswarm(char type, int dimensionnum, int partnum, double w);

clswarm * clinitswarm(char type, int dimensionnum, int partnum, double w);

//swarm particle distribution
void distributeparticles(swarm * school,double * bounds);

void cldistributeparticles(clswarm * school,double * bounds);

//running the swarm
void runswarm(int iterations, swarm  *school, double (*fitness)(double *));

void clrunswarm(int iterations, clswarm *school,double (*fitness)(double *));

//Run the swarm conditionally. The keep_going function should return 0 to keep going and 1 to stop
//inputting 0 into the iterations argument will make the swarm run until the function tells it to stop.
//the double * being passed into keep_going is a array of 5
void conditionalrunswarm(int iterations, swarm *school, double (*fitness)(double *), int (*keep_going)(double *));

void clconditionalrunswarm(int iterations, clswarm *school,double (*fitness)(double *), int (*keep_going)(double *));

//get best solution
//returns school.gbest
double * returnbest(swarm * school);

double * clreturnbest(clswarm * school);

//releasing/ending the swarm
void releaseswarm(swarm * school);

void clreleaseswarm(clswarm * school);

#endif
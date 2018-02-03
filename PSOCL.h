#ifndef PSOCL_H
#define PSOCL_H
/*PSOCL.h
houses structs and function prototypes

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include <CL/cl.h>

//struct for each particle 
typedef struct particle {
    double *present, *pbest, fitness, pfitness,*v;
} particle;

//struct for the swarm
typedef struct swarm {
    int partnum, dimnum;
    double *gbest, gfitness;
    float w;
    particle *school;
} swarm;

//struct for the opencl accelerated swarm
typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, gfitness;
    float w;
    particle *school;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_int ret;
} clswarm;

//swarm initializaion
swarm initswarm(char type, int dimensionnum, int partnum, double w);

clswarm clinitswarm(char type, int dimensionnum, int partnum, double w);

//swarm particle distribution
void distributeparticles(swarm school,double * bounds);

void cldistributeparticles(clswarm school,double * bounds);

//running the swarm
void ruclnswarm(int iterations, clswarm school, double (*fitness)(double *);

void runswarm(int iterations, swarm school,double (*fitness)(double *));

//get best solution
//returns school.gbest
double * returnbest(swarm school);

double * returnbest(clswarm school);

//releasing/ending the swarm
void releaseswarm(swarm school);

void clreleaseswarm(clswarm school);

#endif
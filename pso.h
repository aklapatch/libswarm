#ifndef PSO_H
#define PSO_H
/*pso.h
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

//swarm initializaion
swarm * initswarm(int dimensionnum, int partnum, double w);

//swarm particle distribution
void distributeparticles(swarm * school,double * bounds);

//running the swarm
void runswarm(int iterations, swarm  *school, double (*fitness)(double *));

//Run the swarm conditionally. The keep_going function should return 0 to keep going and 1 to stop
//inputting 0 into the iterations argument will make the swarm run until the function tells it to stop.
//the double * being passed into keep_going is a array of 5
void conditionalrunswarm(int iterations, swarm *school, double (*fitness)(double *), int (*keep_going)(double *));

//get best solution
//returns school.gbest
double * returnbest(swarm * school);

//releasing/ending the swarm
void releaseswarm(swarm * school);

#endif
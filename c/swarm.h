///swarm.h 
/** holds prototypes for c version of PSO
 * Copyright Aaron Klapatch 2018
 */

#ifndef _SWARM_H_
#define _SWARM_H_

#include "error.h"
#include <math.h>
#include <time.h>
#define DEFAULT_DIM 1
#define DEFAULT_PARTNUM 100
#define DEFAULT_W 1
#define C1 1.492
#define C2 2


typedef struct swarm {
    /// no. of particles, no. of dimensions
    int partnum=DEFAULT_PARTNUM,dimnum=DEFAULT_DIM;

    ///best particle dimensions, its fitness, swarm bounds
    double * gbest, *upperbound, *lowerbound;

    ///set that so all fitness numbers will show up
    double gfitness=-HUGE_VAL;

    ///inertial weight and 2 behavioral constants
    float w=DEFAULT_W, c1=C1, c2=C2;

    ///particle data
        double ** presents, **pbests, **v;
        double * pfitnesses, * fitnesses;
} swarm;

///sets no. particles and no. dimensions and w
swarm makeswarm(int, int, float);

///frees swarm data
void freeswarm(swarm );

///sets the number of particles
void setpartnum(int);

///sets no. of dimensions
void setdimnum(int);

///sets inertial weight
void setweight(float);

///sets 2 behavioral constants of the swarm
void setconstants(float, float);

/// sets upper and lower bounds and distributes linearly between them
/** lower bound is first argument, upper bound is second argument */
void distribute(double * , double* );

/// updates (int) number of times with *fitness as a fitness function
void update(int, double (*fitness)(double*) );

///returns the best position in the swarm.
double * getgbest();

///returns the fitness of the best particle
double getgfitness();

#endif
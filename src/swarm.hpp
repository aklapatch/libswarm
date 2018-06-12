/*particle.hpp
houses class and function prototypes for swarm class
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#ifndef _SWARM_HPP_
#define _SWARM_HPP_

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <cstring>

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
        size_t partnum, dimnum;

        ///best particle dimensions, its fitness, swarm bounds
        double * gbest, * upperbound, * lowerbound;

        ///set that so all fitness numbers will show up
        double gfitness;

        ///inertial weight and 2 behavioral constants
        float w, c1, c2;

        ///particle data
        double ** presents, ** pbests, **v;
        double * pfitnesses, * fitnesses;

    public:
        ///defaults to 100 particles and 1 dimension
        swarm();

        ///sets no. particles and no. dimensions and w
        swarm(size_t,size_t,float,float,float);

        ///frees all swarm memory
        ~swarm();

        ///sets the number of particles
        void setPartNum(size_t);

        ///get number of particles
        size_t getPartNum();

        ///sets no. of dimensions
        void setDimNum(size_t);

        ///return dimension number
        size_t getDimNum();

        ///sets inertial weight
        void setWeight(float);

        ///return the inertial weight
        float getWeight();

        ///set the 2 behavior constants
        void setConstants(float, float);

        ///gets 2 behavioral constants of the swarm
        void getConstants(float[2]);

        /// sets upper and lower bounds and distributes linearly between them
        /** lower bound is first argument, upper bound is second argument */
        void distribute(double* , double* );

        /// updates (int) number of times with *fitness as a fitness function
        void update(int, double (*fitness)(double*) );

        ///returns the best position in the swarm.
        void getGBest(double *);

        ///returns the fitness of the best particle
        double getGFitness();

        ///sets particle data
        void setPartData(double **);

        ///returns particle data
        void getPartData(double **);
};

#endif
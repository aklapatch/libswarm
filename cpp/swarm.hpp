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
#include <iostream>
#include <random>

#define DEFAULT_DIM 1
#define DEFAULT_PARTNUM 100
#define DEFAULT_W 1.5

#define C1 1.492
#define C2 2

///there is no particle class, the swarm class has all the data
///that might change later on
class swarm {
    private:
        /// no. of particles, no. of dimensions
        int partnum, dimnum;

        ///best particle dimensions, its fitness, swarm bounds
        std::vector<double> *gbest, *upperbound, *lowerbound;

        double gfitness;

        ///inertial weight
        float w;

        ///particle data
        std::vector<double> *presents, *pbests, *pfitnesses, *fitnesses, *v;


    public:
        ///defaults to 100 particles and 1 dimension
        swarm();

        ///sets no. particles and no. dimensions
        swarm(int, int,float);

        ///frees all swarm memory
        ~swarm();

        ///sets the number of particles
        void setpartnum(int);

        ///sets no. of dimensions
        void setdimnum(int);

        ///sets inertial weight
        void setweight(float);

        /// sets upper and lower bounds and distributes linearly between them
        void distribute(std::vector<double> , std::vector<double> );

        /// updates (int) number of times with *fitness as a fitness function
        void update(int, double (*fitness)(std::vector<double>) );

        ///returns the best position in the swarm.
        std::vector<double> getgbest();
};

#endif

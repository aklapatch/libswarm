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
        int partnum=DEFAULT_PARTNUM, dimnum=DEFAULT_DIM;

        ///best particle dimensions, its fitness, swarm bounds
        double * gbest, * upperbound, * lowerbound;

        ///set that so all fitness numbers will show up
        double gfitness=-HUGE_VAL;

        ///inertial weight and 2 behavioral constants
        float w=DEFAULT_W, c1=C1, c2=C2;

        ///particle data
        std::vector<std::vector<double>> presents, pbests,v;
        std::vector<double> pfitnesses, fitnesses;

    public:
        ///defaults to 100 particles and 1 dimension
        swarm();

        ///sets no. particles and no. dimensions and w
        swarm(int, int,float);

        ///frees all swarm memory
        ~swarm();

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
        void distribute(std::vector<double> , std::vector<double> );

        /// updates (int) number of times with *fitness as a fitness function
        void update(int, double (*fitness)(std::vector<double>) );

        ///returns the best position in the swarm.
        std::vector<double> getgbest();

        ///returns the fitness of the best particle
        double getgfitness();
};

#endif
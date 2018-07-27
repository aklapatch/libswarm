/* swarm.cpp
implementation for swarm
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "Swarm.hpp"
#include <iostream>

void Swarm::Reserve(size_t t_partnum, size_t t_dimnum){

    partnum = t_partnum;
    dimnum = t_dimnum;

    // reserve vector sizes for dimension items
    gbest.resize(dimnum);
    upperbound.resize(dimnum);
    lowerbound.resize(dimnum);

    // resize dim and part oriented items
    presents.resize(dimnum*partnum);
    pbests.resize(dimnum*partnum);
    v.resize(dimnum*partnum);

    //resize sizes for particle oriented items
    pfitnesses.assign(partnum, -HUGE_VAL);
    fitnesses.resize(partnum);
}

//sets dimensions to 1 and number of particles to 100 and w to 1.0
Swarm::Swarm(){

    //set swarm characteristics to defaults
    Reserve(DEFAULT_PARTNUM,DEFAULT_DIM);
    w = DEFAULT_W;
    gfitness=-HUGE_VAL;
    c1=DEFAULT_C1;
    c2=DEFAULT_C2;
}

//sets dimensions to 1 and number of particles to 100 and w to 1.5
Swarm::Swarm(size_t numparts, size_t numdims,float inw, float inc1, float inc2){

    //set swarm characteristics
    Reserve(numparts,numdims);
    w = inw;
    gfitness=-HUGE_VAL;
    c1=inc1;
    c2=inc2;
}

//deallocate all memory
Swarm::~Swarm(){}

//sets number of particles
void Swarm::setPartNum(size_t num){

    Reserve(num,dimnum);
}

//get number of particles
size_t Swarm::getPartNum(){
    return partnum;
}

//sets number of dimensions
void Swarm::setDimNum(size_t num){

    Reserve(partnum,num);
}

//return no. of dimensions
size_t Swarm::getDimNum(){
    return dimnum;
}

//set inertial weight
void Swarm::setWeight(float nw){
    w=nw;
}

float Swarm::getWeight(){
    return w;
}

//set behavioral constants
void Swarm::setC1(float in){
    c1=in;
}

//return constant
float Swarm::getC1(){
    return c1;
}

//set behavioral constants
void Swarm::setC2(float in){
    c2=in;
}

//return constant
float Swarm::getC2(){
    return c2;
}

//distribute particle linearly from lower bound to upper bound
void Swarm::distribute(std::vector<double> lower, std::vector<double> upper){

    //store bounds for later
    upperbound=upper;
    lowerbound=lower;

    std::vector<double> delta(dimnum);
    for(size_t i = 0 ; i < dimnum; ++i){
        delta.at(i) = (upperbound[i] - lowerbound[i])/(partnum-1);
    }

    for(size_t j = 0; j < partnum; ++j){
        for(size_t k = 0; k < dimnum; ++k){
            presents.at(j*dimnum+k) = j*delta.at(k);
        }
    }
}

// XORshift random number generator from 
// https://codingforspeed.com/using-faster-psudo-random-generator-xorshift/
size_t rng(){
    static size_t x = 123456789;
    static size_t y = 362436069;
    static size_t z = 521288629;
    static size_t w = 88675123;
    size_t t;
    t = x ^ (x << 11);   
    x = y; 
    y = z; 
    z = w;   
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

//run the position and velocity update equation
void Swarm::update(int times, double (*fitness) (std::vector<double>, size_t)){

    while(times--){
        for(size_t i=0;i<partnum;++i){

            //get fitness
            fitnesses[i] = fitness(presents,i);

            //if the fitness is better than the particle best store the best position
            if(fitnesses[i]>pfitnesses[i]){

                //set new fitness
                pfitnesses[i]=fitnesses[i];

                //store position
                for(size_t j = 0; j<dimnum; ++j)
                    pbests[i*dimnum+j]=presents[i*dimnum+j];

                //if fitness is better than global fitness
                if(fitnesses[i]>gfitness){

                    //set new fitness
                    gfitness=fitnesses[i];

                    //store best position
                    for(size_t j=0;j<dimnum;++j)
                        gbest[j]= presents[i*dimnum + j];
                }
            }

            for(size_t j=0;j<dimnum;++j){

                //update velocity
                v[i*dimnum+j]=w*v[i*dimnum+j] + c1*RAN*(pbests[i*dimnum+j]-presents[i*dimnum+j]) +c2*RAN*(gbest[j]-presents[i*dimnum+j]);

                //update position
                presents[i*dimnum+j]=presents[i*dimnum+j]+v[i*dimnum+j];

                //if it exceeds the bounds
                if(presents[i*dimnum+j]>upperbound[j]){
                    presents[i*dimnum+j]=upperbound[j];

                //if it goes below the lower bound
                } else if (presents[i*dimnum+j]< lowerbound[j]){
                    presents[i*dimnum+j]=lowerbound[j];
                }
            }
        }
    }

    // do one final check for the fitness
    for(size_t i=0;i<partnum;++i){

        //get fitness
        fitnesses[i] = fitness(presents,i);

        //if the fitness is better than the particle best store the best position
        if(fitnesses[i]>pfitnesses[i]){

            //set new fitness
            pfitnesses[i]=fitnesses[i];

            //store position
            for(size_t j = 0; j<dimnum; ++j)
                pbests[i*dimnum+j]=presents[i*dimnum+j];

            //if fitness is better than global fitness
            if(fitnesses[i]>gfitness){

                //set new fitness
                gfitness=fitnesses[i];

                //store best position
                for(size_t j=0;j<dimnum;++j)
                    gbest[j]= presents[i*dimnum + j];
            }
        }
    }
}

//returns best position of the swarm
std::vector<double> Swarm::getGBest(){
    return gbest;
}

//returns the fitness of the best particle
double Swarm::getGFitness(){
    return gfitness;
}

//copies input array data to particle array
void Swarm::setPartData(std::vector<double> t_partdata){
    std::copy(t_partdata.data(), t_partdata.data() + partnum, presents.begin());
}

//copy particle data into input argument
std::vector<double> Swarm::getPartData(){
    return presents;
}

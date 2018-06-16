/* swarm.cpp
implementation for swarm
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "swarm.hpp"
#include <iostream>

//sets dimensions to 1 and number of particles to 100 and w to 1.0
Swarm::Swarm(){

    //set swarm characteristics to defaults
    partnum=DEFAULT_PARTNUM;
    dimnum=DEFAULT_DIM;
    w = DEFAULT_W;
    gfitness=-HUGE_VAL;
    c1=C1;
    c2=C2;

    try {
        //set all vector sizes to default sizes
        gbest = new double[DEFAULT_DIM];

        pfitnesses = new double [DEFAULT_PARTNUM];
        fitnesses= new double [DEFAULT_PARTNUM];

        pbests= new double * [DEFAULT_PARTNUM];
        presents= new double * [DEFAULT_PARTNUM];
        v= new double * [DEFAULT_PARTNUM];

        //set all vectors to proper dimensions
        int i;
        for(i=0;i<DEFAULT_PARTNUM;++i){
            presents[i]= new double [DEFAULT_DIM];
            pbests[i]= new double [DEFAULT_DIM];
            v[i]= new double [DEFAULT_DIM];
            pfitnesses[i]=-HUGE_VAL;
        }
    } catch (std::bad_alloc& ac) {
        std::cerr << "Memory allocation failed: "<<ac.what() <<std::endl;
        exit(1);
    }
}

//sets dimensions to 1 and number of particles to 100 and w to 1.5
Swarm::Swarm(size_t numparts, size_t numdims,float inw, float inc1, float inc2){

    //set swarm characteristics
    partnum=numparts;
    dimnum=numdims;
    w = inw;
    gfitness=-HUGE_VAL;
    c1=inc1;
    c2=inc2;

    try {
        //set all vector sizes
        gbest = new double[dimnum];

        pfitnesses = new double [partnum];
        fitnesses= new double [partnum];

        pbests= new double * [partnum];
        presents= new double * [partnum];
        v= new double * [partnum];

        //set all vectors to proper dimensions
        while(numparts--){
            presents[numparts]= new double [dimnum];
            pbests[numparts]= new double [dimnum];
            v[numparts]= new double [dimnum];
            pfitnesses[numparts]=-HUGE_VAL;
        }
    } catch (std::bad_alloc& ac) {
        std::cerr << "Memory allocation failed: "<<ac.what() <<std::endl;
        exit(1);
    }
}

//deallocate all memory
Swarm::~Swarm(){

    delete [] gbest;
    delete [] pfitnesses;
    delete [] fitnesses;

    while(partnum--){
        delete [] presents[partnum];
        delete [] pbests[partnum];
        delete [] v[partnum];
    }

    delete [] pbests;
    delete [] presents;
    delete [] v;
}

//sets number of particles
void Swarm::setPartNum(size_t num){

    delete [] pfitnesses;
    delete [] fitnesses;

    while(partnum--){
        delete [] presents[partnum];
        delete [] pbests[partnum];
        delete [] v[partnum];
    }

    delete [] pbests;
    delete [] presents;
    delete [] v;

    try {
        //reset particle swarm #
        partnum=num;

        pfitnesses = new double [num];
        fitnesses= new double [num];

        pbests= new double * [num];
        presents= new double * [num];
        v= new double * [num];

        //set all vectors to proper dimensions
        while(num--){
            presents[num]= new double [dimnum];
            pbests[num]= new double [dimnum];
            v[num]= new double [dimnum];
            pfitnesses[num]=-HUGE_VAL;
        }
    } catch (std::bad_alloc& ac) {
        std::cerr << "Memory allocation failed: "<<ac.what() <<std::endl;
        exit(1);
    }
}

//get number of particles
size_t Swarm::getPartNum(){
    return partnum;
}

//sets number of dimensions
void Swarm::setDimNum(size_t num){

    dimnum=num;

    try {

        delete [] gbest;
        gbest = new double [num];

        int i;
        for(i=0;i<partnum;++i){
            delete [] presents[i];
            delete [] pbests[i];
            delete [] v[i];

            presents[i] = new double [num];
            pbests[i] = new double [num];
            v[i] = new double [num];
        }
    } catch (std::bad_alloc& ac) {
        std::cerr << "Memory allocation failed: "<<ac.what() <<std::endl;
        exit(1);
    }
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
void Swarm::distribute(double * lower, double * upper){

    //store bounds for later
    upperbound=upper;
    lowerbound=lower;

    size_t i,j;

    //allocate memory for and distribute particles
    try{
        double * delta = new double [dimnum];

        for(i=0; i<dimnum; ++i){
            delta[i]=(upperbound[i] - lowerbound[i])/(partnum-1);
            gbest[i]=0;

            for(j=0;j<partnum;++j){
                presents[j][i]=j*delta[i] + lowerbound[i];
                pbests[j][i]=0;
                v[j][i]=0;
            }
        }

        delete [] delta;

    } catch (std::bad_alloc& ac) {
        std::cerr << "Memory allocation failed: "<<ac.what() <<std::endl;
        exit(1);
    }
}

//run the position and velocity update equation
void Swarm::update(int times, double (*fitness) (double*)){

    int i,j;

    //make random number generator C++11
    std::random_device gen;
    std:: uniform_real_distribution<double> distr(1,0);

    while(times--){
        for(i=0;i<partnum;++i){

            //get fitness
            fitnesses[i] = fitness(presents[i]);

            //if the fitness is better than the particle best store the best position
            if(fitnesses[i]>pfitnesses[i]){

                //set new fitness
                pfitnesses[i]=fitnesses[i];

                //store position
                for(j=0;j<dimnum;++j)
                    pbests[i][j]=presents[i][j];

                //if fitness is better than global fitness
                if(fitnesses[i]>gfitness){

                    //set new fitness
                    gfitness=fitnesses[i];

                    //store best position
                    for(j=0;j<dimnum;++j)
                        gbest[j]=presents[i][j];
                }
            }

            for(j=0;j<dimnum;++j){

                //update velocity
                v[i][j]=w*v[i][j] + c1*distr(gen)*(pbests[i][j]-presents[i][j]) +c2*distr(gen)*(gbest[j]-presents[i][j]);

                //update position
                presents[i][j]=presents[i][j]+v[i][j];

                //if it exceeds the bounds
                if(presents[i][j]>upperbound[j]){
                    presents[i][j]=upperbound[j];

                //if it goes below the lower bound
                } else if (presents[i][j]< lowerbound[j]){
                    presents[i][j]=lowerbound[j];
                }
            }
        }
    }

    // do one final check for the fitness
    for(i=0;i<partnum;++i){
        //get fitness
        fitnesses[i] = fitness(presents[i]);

        //if the fitness is better than the particle best store the best position
        if(fitnesses[i]>pfitnesses[i]){

            //set new fitness
            pfitnesses[i]=fitnesses[i];

            //store position
            for(j=0;j<dimnum;++j)
                pbests[i][j]=presents[i][j];

            //if fitness is better than global fitness
            if(fitnesses[i]>gfitness){

                //set new fitness
                gfitness=fitnesses[i];

                //store best position
                for(j=0;j<dimnum;++j)
                    gbest[j]=presents[i][j];
            }
        }
    }
}

//returns best position of the swarm
void Swarm::getGBest(double * out){
    size_t i=dimnum;
    while(i--)
        out[i]=gbest[i];
}

//returns the fitness of the best particle
double Swarm::getGFitness(){
    return gfitness;
}

//copies input array data to particle array
void Swarm::setPartData(double ** in){
    size_t j, i=partnum;
    while (i--){
        for(j=0;j<dimnum;++j){
            presents[i][j]=in[i][j];
        }
    }
}

//copy particle data into input argument
void Swarm::getPartData(double ** out){

    size_t i=partnum;
    while(i--)
        memcpy(out[i],presents[i],sizeof(double)*dimnum);
}

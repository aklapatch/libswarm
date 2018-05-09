/*swarm.cpp
implementation for swarm
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "swarm.hpp"
#include <iostream>

///sets dimensions to 1 and number of particles to 100 and w to 1.0
swarm::swarm(){

    ///set swarm characteristics to defaults
    partnum=DEFAULT_PARTNUM;
    dimnum=DEFAULT_DIM;
    w = DEFAULT_W;
    gfitness=-HUGE_VAL; 
    c1=C1;
    c2=C2;   
  
    try {
        ///set all vector sizes to default sizes
        gbest = new double[DEFAULT_DIM];
    
        pfitnesses = new double [DEFAULT_PARTNUM];
        fitnesses= new double [DEFAULT_PARTNUM];
    
        pbests= new double * [DEFAULT_PARTNUM];
        presents= new double * [DEFAULT_PARTNUM];    
        v= new double * [DEFAULT_PARTNUM];

        ///set all vectors to proper dimensions
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

///sets dimensions to 1 and number of particles to 100 and w to 1.5
swarm::swarm(int numparts, int numdims,float inw, float c1in, float c2in){

    ///set swarm characteristics to defaults
    partnum=numparts;
    dimnum=numdims;
    w = DEFAULT_W;
    gfitness=-HUGE_VAL;   
    c1=c1in;
    c2=c2in; 
  
    try {
        ///set all vector sizes to default sizes
        gbest = new double[dimnum];
        
        pfitnesses = new double [numparts];
        fitnesses= new double [numparts];
        
        pbests= new double * [numparts];
        presents= new double * [numparts];    
        v= new double * [numparts];

        ///set all vectors to proper dimensions
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

///deallocate all memory
swarm::~swarm(){
    
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

///sets number of particles
void swarm::setPartNum(int num){
    
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
        ///reset particle swarm #
        partnum=num;

        pfitnesses = new double [num];
        fitnesses= new double [num];
        
        pbests= new double * [num];
        presents= new double * [num];    
        v= new double * [num];

        ///set all vectors to proper dimensions
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

///get number of particles
int swarm::getPartNum(){
    return partnum;
}

///sets number of dimensions
void swarm::setDimNum(int num){
    
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

///return no. of dimensions
unsigned int swarm::getDimNum(){
    return dimnum;
}

///set inertial weight
void swarm::setWeight(float nw){
    w=nw;
}

float swarm::getWeight(){
    return w;
}

///set behavioral constants
void swarm::setConstants(float nc1,float nc2){
    c1=nc1;
    c2=nc2;
}

///return constant array
float * swarm::getConstants(){
    float * out=new float[2];
    out[0]=c1;
    out[1]=c2;
    return out;
}

///distribute particle linearly from lower bound to upper bound
void swarm::distribute(double * lower, double * upper){
    
    ///store bounds for later
    upperbound=upper;
    lowerbound=lower;
    
    int i,j;
    
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

///run the position and velocity update equation
void swarm::update(int times, double (*fitness) (double*)){
        
    int i,j;

    ///make random number generator C++11
    std::random_device gen;
    std:: uniform_real_distribution<double> distr(1,0);

    while(times--){
        for(i=0;i<partnum;++i){
            for(j=0;j<dimnum;++j){

                ///update velocity                
                v[i][j]=w*v[i][j] + c1*distr(gen)*(pbests[i][j]-presents[i][j]) +c2*distr(gen)*(gbest[j]-presents[i][j]);

                ///update position
                presents[i][j]=presents[i][j]+v[i][j];

                ///if it exceeds the bounds
                if(presents[i][j]>upperbound[j]){
                    presents[i][j]=upperbound[j];

                ///if it goes below the lower bound
                } else if (presents[i][j]< lowerbound[j]){
                    presents[i][j]=lowerbound[j];
                }
            }

            ///get fitness
            fitnesses[i] = fitness(presents[i]);

            ///if the fitness is better than the particle best store the best position
            if(fitnesses[i]>pfitnesses[i]){
                
                ///set new fitness
                pfitnesses[i]=fitnesses[i];
                
                ///store position
                for(j=0;j<dimnum;++j){
                    pbests[i][j]=presents[i][j];
                }

                ///if fitness is better than global fitness
                if(fitnesses[i]>gfitness){
                    
                    ///set new fitness
                    gfitness=fitnesses[i];

                    ///store best position
                    for(j=0;j<dimnum;++j){
                        gbest[j]=presents[i][j];
                    }
                }
            }
        }
    }
}

///returns best position of the swarm
double * swarm::getGBest(){
    return gbest;
}   

///returns the fitness of the best particle
double swarm::getGFitness(){
    return gfitness;
}

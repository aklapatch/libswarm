/*swarm.cpp
implementation for swarm
for openclless version

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "swarm.hpp"

///sets dimensions to 1 and number of particles to 100 and w to 1.5
swarm::swarm(){

    ///set swarm characteristics to defaults
    partnum=DEFAULT_PARTNUM;
    dimnum=DEFAULT_DIM;
    w = DEFAULT_W;

    
  
    ///set all vector sizes to default sizes
    gbest.resize(DEFAULT_DIM);
    
    pfitnesses.resize(DEFAULT_PARTNUM);
    fitnesses.resize(DEFAULT_PARTNUM);
    
    pbests.resize(DEFAULT_PARTNUM);
    presents.resize(DEFAULT_PARTNUM);    
    v.resize(DEFAULT_PARTNUM);


  
    int i;
    for(i=0;i<DEFAULT_PARTNUM;++i){
        presents[i].resize(DEFAULT_DIM);
        pbests[i].resize(DEFAULT_DIM);
        v[i].resize(DEFAULT_DIM);
    }
}

///sets dimensions to 1 and number of particles to 100 and w to 1.5
swarm::swarm(int numdims, int numparts,float inw){

    ///set swarm characteristics to defaults
    partnum=numparts;
    dimnum=numdims;
    w= inw;

    ///set all vector sizes to default sizes
    gbest.resize(dimnum);
    
    pfitnesses.resize(numparts);
    fitnesses.resize(numparts);
    
    pbests.resize(numparts);
    presents.resize(numparts);    
    v.resize(numparts);

    ///set all arrays to the proper size
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
        v[i].resize(dimnum);
    }
}

swarm::~swarm(){

}

void swarm::setpartnum(int num){
    ///reset particle swarm #
    partnum=num;
    
    ///resize all internal vectors
    pfitnesses.resize(partnum);
    fitnesses.resize(partnum);
    pbests.resize(partnum);
    presents.resize(partnum);    
    v.resize(partnum);
    
    ///ensure all contained vectors are the right size
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
        v[i].resize(dimnum);
    }
}

void swarm::setdimnum(int num){
    dimnum=num;
    
    gbest.resize(dimnum);
    
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
        v[i].resize(dimnum);
    }
    
}

void swarm::setweight(float nw){
    w=nw;
}

void swarm::setconstants(float nc1,float nc2){
    c1=nc1;
    c2=nc2;
}


void swarm::distribute(std::vector<double> lower, std::vector<double> upper){
    
    ///store bounds for later
    upperbound=upper;
    lowerbound=lower;
    
    int i,j;
    
    std::vector<double> delta;
    delta.resize(dimnum);
    
    for(i=0; i<dimnum; ++i){
        delta[i]=(upperbound[i] - lowerbound[i])/(partnum-1);
        gbest[i]=0;
        
        for(j=0;j<partnum;++j){
            presents[j][i]=j*delta[i] + lowerbound[i];
            pbests[i][j]=0;
            v[j][i]=0;
        }
    }
}

void swarm::update(int times, double (*fitness) (std::vector<double>)){
        
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

            fitnesses[i] = fitness(presents[i]);


            if(fitnesses[i]>pfitnesses[i]){
                pfitnesses[i]=fitnesses[i];

                for(j=0;j<dimnum;++j){
                    pbests[i][j]=presents[i][j];
                }

                if(fitnesses[i]>gfitness){
                    gfitness=fitnesses[i];

                    for(j=0;j<dimnum;++j){
                    gbest[j]=presents[i][j];
                    }
                }

            }
        }
    }
}

///returns best position of the swarm
std::vector<double> swarm::getgbest(){
    return gbest;
}   

///returns the fitness of the best particle
double swarm::getgfitness(){
    return gfitness;
}

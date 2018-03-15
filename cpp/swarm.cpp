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
    w= DEFAULT_W;

    
    ///get the gbest coordinates for the swarm
    gbest= new std::vector<double>;
    gbest->resize(DEFAULT_DIM);

    ///get the particle fitness array for the swarm and set its size
    pfitnesses= new std::vector<double>;
    pfitnesses->resize(DEFAULT_PARTNUM);

    ///get the finess array for the swarm and set its size
    fitnesses= new std::vector<double>;
    fitnesses->resize(DEFAULT_PARTNUM);

    ///get the particle position and best coordinates for the swarm
    presents= new std::vector<double>[DEFAULT_PARTNUM];
    pbests= new std::vector<double>[DEFAULT_PARTNUM];

    ///get a v for each particle
    v=new std::vector<double>[DEFAULT_PARTNUM];
    
    ///set all arrays to the proper size
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

    
    ///get the gbest coordinates for the swarm
    gbest= new std::vector<double>;
    gbest->resize(dimnum);

    ///get the particle fitness array for the swarm and set its size
    pfitnesses= new std::vector<double>;
    pfitnesses->resize(partnum);

    ///get the finess array for the swarm and set its size
    fitnesses= new std::vector<double>;
    fitnesses->resize(partnum);

    ///get the particle position and best coordinates for the swarm
    presents= new std::vector<double>[partnum];
    pbests= new std::vector<double>[partnum];

    ///get a v for each particle
    v=new std::vector<double>[dimnum];
    
    ///set all arrays to the proper size
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
        v[i].resize(dimnum);
    }
}

swarm::~swarm(){
    delete gbest;
    delete pfitnesses;
    delete fitnesses;
    delete [] presents;
    delete [] pbests;
    delete [] v;
}

void swarm::setpartnum(int num){
    partnum=num;
    
    pfitnesses->resize(partnum);
    fitnesses->resize(partnum);
    
    delete [] presents;
    delete [] pbests;
    delete [] v;
    
    presents= new std::vector<double>[partnum];
    pbests= new std::vector<double>[partnum];
    v=new std::vector<double>[dimnum];
    
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
        v[i].resize(dimnum);
    }
}

void swarm::setdimnum(int num){
    dimnum=num;
    
    gbest->resize(dimnum);
    
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

void swarm::distribute(std::vector<double> lower, std::vector<double> upper){
    
    upperbound= new std::vector<double>;
    *upperbound=upper;
    
    lowerbound=new std::vector<double>;
    *lowerbound=lower;
    
    int i,j;
    
    double * delta= new double [dimnum];
    
    for(i=0; i<dimnum; ++i){
        delta[i]=(upperbound[0][i] - lowerbound[0][i])/(partnum-1);
        gbest[0][i]=0;
        
        for(j=0;j<partnum;++j){
            presents[j][i]=j*delta[i] + lowerbound[0][i];
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
                v[i][j]=w*v[i][j] + C1*distr(gen)*(pbests[i][j]-presents[i][j]) +C2*distr(gen)*(gbest[0][j]-presents[i][j]);

                ///update position
                presents[i][j]=presents[i][j]+v[i][j];

            }

            fitnesses[0][i] = fitness(presents[i]);


            if(fitnesses[0][i]>pfitnesses[0][i]){
                pfitnesses[0][i]=fitnesses[0][i];

                for(j=0;j<dimnum;++j){
                    pbests[i][j]=presents[i][j];
                }

                if(fitnesses[0][i]>gfitness){
                    gfitness=fitnesses[0][i];

                    for(j=0;j<dimnum;++j){
                    gbest[0][j]=presents[i][j];
                    }
                }

            }
        }
    }
}

std::vector<double> swarm::getgbest(){
    return gbest[0];
}   


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
    pfitnesses->resize(DEFAULT_DIM);

    ///get the finess array for the swarm and set its size
    fitnesses= new std::vector<double>;
    pfitnesses->resize(DEFAULT_DIM);

    ///get the particle position and best coordinates for the swarm
    presents= new std::vector<double>[DEFAULT_PARTNUM];
    pbests= new std::vector<double>[DEFAULT_PARTNUM];
    
    ///set all arrays to the proper size
    int i;
    for(i=0;i<DEFAULT_PARTNUM;++i){
        presents[i].resize(DEFAULT_DIM);
        pbests[i].resize(DEFAULT_DIM);
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
    pfitnesses->resize(dimnum);

    ///get the finess array for the swarm and set its size
    fitnesses= new std::vector<double>;
    pfitnesses->resize(dimnum);

    ///get the particle position and best coordinates for the swarm
    presents= new std::vector<double>[partnum];
    pbests= new std::vector<double>[partnum];
    
    ///set all arrays to the proper size
    int i;
    for(i=0;i<partnum;++i){
        presents[i].resize(dimnum);
        pbests[i].resize(dimnum);
    }
}

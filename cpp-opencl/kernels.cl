///kernels.cl
/** houses all kernels for this project */

#define id(x) (get_global_id(x))

int get_global_id(int);

///return the index with the biggest number
int sort(float * array,int size){
    int ret=0;
    float biggest=-HUGE_VALF;
    while(size-->1){
        if(array[size]>biggest){
            biggest=array[size];
            ret=size;
        }
    }
    return ret;
}

__kernel void compare( __global float *presents,
                        __global float * gbest,
                        __global float * fitnesses,
                        __global float * gfitness,
                        __global int * partnum,
                        __global int * dimnum) {
    
    ///copy most fit particle into the gbest array
    int i,index=sort(fitnesses,*partnum);
    if(fitnesses[index]>*gfitness){
        for(i=0;i<(*dimnum);++i){
            gbest[i]=presents[index*(*partnum)+i];
        }
    }
}

__kernel void distribute(__global float * lowerbound, 
                         __global float * upperbound,
                         __global float * delta,
                         __global float * presents,
                         __global float * pbests,
                         __global int * partnum){
    unsigned int dex[3] = {id(1), id(0)*(*partnum) +id(1), id(0)};

    ///id(1) is dimension number, id(0) is particle number
    delta[dex[0]]=(upperbound[dex[0]]-lowerbound[dex[0]])/(*partnum);

    ///distribute the particle between the upper and lower boundaries linearly
    presents[dex[1]]=dex[2]*delta[dex[0]] + lowerbound[dex[0]];
    pbests[dex[1]]=0;
}


#define fitness(x) x[index]

__kernel void update( __global float * presents,
                      __global float * v,
                      __global float * w, 
                      __global float * rand,
                      __global float * pfitnesses,
                      __global float *upperbound,
                      __global float * pbest, 
                      __global float * gbest, 
                      __global float * lowerbound,
                      __global float * fitnesses, 
                      __global int * partnum) {
    int index= id(0)*(*partnum)+id(1);

    ///id(0) is partnum, id(1) is dimiension number
    //velocity update
    v[index]=*w*v[index]
     + rand[index]*(pbest[index]- presents[index])
     + rand[index+1]*(gbest[id(1)]-presents[index]);

    //position update
    presents[index]=presents[index]+v[index];
                
    //upper bound check
    if(presents[index]>upperbound[id(1)]){
        presents[index]=upperbound[id(1)];

    ///lower bounds check
    } else if(presents[index]<lowerbound[id(1)]){
        presents[index]=lowerbound[id(1)];
    }
          
    //evaluating how fit the particle is with passed function
    fitnesses[id(0)]= fitness(presents);
}

///compares and copies a coordinates into a pbest if necessary
__kernel void update2(__global float * fitnesses,
                      __global int * dimnum,
                      __global float * pfitnesses,
                      __global float * presents,
                      __global float * pbest,
                      __global int * partnum){
    int j=get_global_id(0);
    ///if the fitness is better than the pfitness, copy the values to pbest array
    if(fitnesses[j]>pfitnesses[j]){
        
        int i=*dimnum;
        while(i--){
            pbest[j*(*partnum)+i]=presents[j*(*partnum)+i];
        }
    }
}

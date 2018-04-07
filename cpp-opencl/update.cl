///update.cl
/**the PSO update equation run in gpu memory */


#define id(x) (get_global_id(x))

#define fitness(x) x[index]

int get_global_id(int);

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
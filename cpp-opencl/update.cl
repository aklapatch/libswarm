///update.cl
/**the PSO update equation run in gpu memory */


#define id(x) (get_global_id(x))

#define fitness(x) -(x[id(0)*partnum+ id(1)]*x[id(0)*partnum+ id(1)])

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

    ///id(0) is partnum, id(1) is dimiension number
    //velocity update
    v[id(0)*partnum +id(1)]=*w*v[id(0)*partnum +id(1)]
     + rand[id(0)*partnum +id(1)]*(pbest[id(0)*partnum +id(1)]- presents[id(0)*partnum +id(1)])
     + rand[id(0)*partnum +id(1)+1]*(gbest[id(1)]-presents[id(0)*partnum +id(1)]);

    //position update
    presents[id(0)*partnum +id(1)]=presents[id(0)*partnum +id(1)]+v[id(0)*partnum +id(1)];
                
    //upper bound check
    if(presents[id(0)*partnum +id(1)]>upperbound[id(1)]){
        presents[id(0)*partnum +id(1)]=upperbound[id(1)];

    ///lower bounds check
    } else if(presents[id(0)*partnum +id(1)]<lowerbound[id(1)]){
        presents[id(0)*partnum +id(1)]=lowerbound[id(1)];
    }
          
    //evaluating how fit the particle is with passed function
    fitnesses[id(0)]= fitness(presents);
}

///compares and copies a good 
__kernel void update2(__global float * fitnesses,
                      __global int * dimnum,
                      __global float * pfitnesses){
    int j=get_global_id(0);
    ///if the fitness is better than the pfitness, copy the values to pbest array
    if(fitnesses[j]>pfitnesses[j]){
        
        int i=*dimnum;
        while(i--){
            pbest[i]=presents[j*partnum+i];
        }
    }
}
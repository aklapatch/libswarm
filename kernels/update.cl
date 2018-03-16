///update.cl
/**the PSO update equation run in gpu memory */


#define id(x) (get_global_id(x))

#define fitness(x) -(x[id(0)*partnum+ id(1)]*x[id(0)*partnum+ id(1)])

int get_global_id(int);

__kernel void psoadd( __global float * presents,
                      __global float * v,
                      float w, 
                      __global float * rand,
                      __global float * pfitnesses,
                      __global float *upperbound,
					  __global float * gfitness,
                      __global float * pbest, 
                      __global float * gbest, 
                      __global float * lowerbound,
                      __global float * fitnesses, 
                      int dimnum, 
                      int partnum) {

    ///id(0) is partnum, id(1) is dimiension number
    //velocity update
    v[id(0)*partnum +id(1)]=w*v[id(0)*partnum +id(1)]
     + rand[id(1)]*(pbest[id(0)*partnum +id(1)]- presents[id(0)*partnum +id(1)])
     + rand[id(1)+1]*(gbest[id(1)]-presents[id(0)*partnum +id(1)]);

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
            
    //setting particle's best position based on fitness
    if(fitnesses[id(0)]>pfitnesses[id(0)]) {
        int fr;
        pfitnesses[id(0)] = fitnesses[id(0)];

        for(fr=0;fr<dimnum;++fr){
            pbest[fr]=presents[id(0)*partnum +fr];
        }

        //setting new best particle in the swarm
        if(fitnesses[id(0)]>(*gfitness)){
            *gfitness=fitnesses[id(0)];   

            for(fr=0;fr<dimnum;++fr){
                gbest[fr]=presents[id(0)*partnum +fr];
            }
        }
    }    
}
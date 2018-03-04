/*psocl.cl
Houses the main function to add the particle positions.
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#define fitness(x) ((x[0])*(x[0]))
    
int get_global_id(int);

__kernel void getdelta(__global float * delta, __global float * bounds, int partnum){
    int i = 2*get_global_id(0);

    if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
		delta[i/2]=(float)((bounds[i+1]-bounds[i])/partnum);    
    }
    else{   //if the first bound is higher than the next
        delta[i/2]=(float)((bounds[i]-bounds[i+1])/partnum);
    }
}
__kernel void distrparts(__global int partnum,__global int dimnum, float * present, ,__global float * bounds, float * delta){
    int i=2*get_global_id(0);	/// 2* particle dimensions
	int j = get_global_id(1);	/// number of particles

    if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
			
            present[j][i/2]=bounds[i]+partnum*delta;
    }
    else{   //if the first bound is higher than the next
            present[j][i/2]=bounds[i+1]+partnum*delta;
    } 
}

__kernel void psoadd( __global float ** present, __global float ** v, __global float w, __global float ** rand, __global float * pfitness, 
						__global float * gfitness, __global ** pbest, __global * gbest, __global float * bounds, global * fitness) {
    int i=get_global_id(0);		///< the number of particles
    int j=2*get_global_id(1); ///the number of dimensions
    int fr=0;

    //velocity update
    v[i][j/2]=w*v[i][j/2])
     + rand[j]*(pbest[i][j/2]- present[i][j/2])
    + rand[j+1]*(gbest[j/2]-present[i][j/2]);

    //position update
    present[i][j/2]=present[i][j/2]+v[i][j/2];
                
    //upper bound check (intolerant of which bound is which)
    if(present[i][j/2]>((bounds[j]>bounds[j+1])?bounds[j]:bounds[j+1])){
        present[i][j/2]=(bounds[j]>bounds[j+1])?bounds[j]:bounds[j+1];
    }

    //lower bound check (intolerant of which bound is which)
    else if(present[i][j/2]<((bounds[j]<bounds[j+1])?bounds[j]:bounds[j+1])){
        present[i][j/2]=(bounds[j]<bounds[j+1])?bounds[j]:bounds[j+1];
    }
          
    //evaluating how fit the particle is with passed function
    fitness= fitness(present[i]);
            
    //setting particle's best position based on fitness
    if(fitness[i]>herd[i].pfitness){
        pfitness[i] = fitness[i];
        for(fr=0;fr<dimnum;++fr){
            pbest[i][fr]=present[i][fr];
        }
    }

    //setting new best particle in the swarm
    //this might be moved inside the previous if statement for optimization
    if(fitness[i]>gfitness){
        gfitness=fitness[i];                
        for(fr=0;fr<dimnum;++fr){
            pbest[i][fr]=present[i][fr];
        }
    }
}



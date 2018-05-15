///kernels.cl
/** houses all kernels for this project */

///include fitness function
#include "fitness.c"

///return the index with the biggest number
int sort(__global float * array,int size){
	int out=0;
	float biggest=-INFINITY;
	while(size--){
		if(array[size]>biggest){
			biggest=array[size];
			out=size;
		}
	}
	return out;
}

__kernel void compare( __global float *presents,
						__global float * gbest,
						__global float * fitnesses,
						__global float * gfitness,
						__constant int * partnum,
						__constant int * dimnum ) {

	///copy most fit particle into the gbest array
	int i=dimnum[0],index=sort(fitnesses,*partnum);

	if(fitnesses[index]>gfitness[0]) {

		///copy the array fitness
		gfitness[0]=fitnesses[index];

		///copy array into gbest array
		while(i--) {
			gbest[i]=presents[index*dimnum[0]+i];
		}
	}
}

__kernel void distribute(__global float * lowerbound,
						 __global float * upperbound,
						 __global float * delta,
						 __global float * presents,
						 __global float * pbests,
						 __constant int * dimnum,
						 __constant int * partnum){
	unsigned int dex[3]={get_global_id(1), get_global_id(0)*(*dimnum) +get_global_id(1), get_global_id(0)} ;

	///get_global_id(1) is dimension number, get_global_id(0) is particle number
	delta[dex[0]]=(upperbound[dex[0]]-lowerbound[dex[0]])/(partnum[0]-1);

	///distribute the particle between the upper and lower boundaries linearly
	presents[dex[1]]=dex[2]*delta[dex[0]] + lowerbound[dex[0]];
	pbests[dex[1]]=0;
}

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
					  __constant int * dimnum,
					  __constant int * partnum,
					  __constant float * c1,
					  __constant float * c2) {
					  
	int index= get_global_id(0)*(*dimnum)+get_global_id(1);
	int randex = index*2;
	int dex0=get_global_id(0);
	int dex1=get_global_id(1);

	///get_global_id(0) is partnum, get_global_id(1) is dimiension number
	//velocity update
	v[index]=(*w)*v[index]
	 + c1[0]*rand[randex]*(pbest[index]- presents[index])
	 + c2[0]*rand[randex+1]*(gbest[dex1]-presents[index]);

	//position update
	presents[index]=presents[index]+v[index];

	//upper bound check
	if(presents[index]>upperbound[dex1]){
		presents[index]=upperbound[dex1];

	///lower bounds check
	} else if(presents[index]<lowerbound[dex1]){
		presents[index]=lowerbound[dex1];
	}
}

///compares and copies a coordinates into a pbest if necessary
__kernel void update2(__global float * fitnesses,
						__global int * dimnum,
						__global float * pfitnesses,
						__global float * presents,
						__global float * pbest,
						__constant int * partnum ) {

	const unsigned int j=get_global_id(0);
	unsigned int offset=j*dimnum[0];

	///evaluate fitness of the particle
	/** fitness function is in fitness.c */
	fitnesses[j]=fitness(presents,offset,*dimnum);

	///if the fitness is better than the pfitness, copy the values to pbest array
	if(fitnesses[j]>pfitnesses[j]) {

		///copy new fitness
		pfitnesses[j]=fitnesses[j];

		unsigned int i=dimnum[0];

		while(i--) {
			pbest[offset+i]=presents[offset+i];
		}
	}
}
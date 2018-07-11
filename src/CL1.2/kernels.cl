//kernels.cl
/** houses all kernels for this project */

//include fitness function
#include "fitness.cl"

//return the index with the biggest number
int sort(__global float * array,unsigned int size){
	int out=0;
	unsigned int i = size;
	float biggest=-INFINITY;
	while(i-->0){
		if(array[i]>biggest){
			biggest=array[i];
			out=i;
		}
	}
	return out;
}

__kernel void compare( __global float *presents,
						__global float * gbest,
						__global float * fitnesses,
						__global float * gfitness,
						unsigned int partnum,
						unsigned int dimnum) {

	//copy most fit particle into the gbest array
	unsigned int i=dimnum;
	unsigned int index=sort(fitnesses,partnum);

	if(fitnesses[index] > *gfitness) {

		//copy the array fitness
		*gfitness=fitnesses[index];

		//copy array into gbest array
		while(i--) 
			gbest[i]=presents[index*dimnum+i];
	}
}

//TEST FUNCTION
//get the delta array
__kernel void getDelta(__global float * lowerbound,
					__global float * upperbound,
					__global float * delta,
					unsigned int partnum) {
	//compute the delta
	unsigned int i=get_global_id(0);
	delta[i]=(upperbound[i]-lowerbound[i])/(partnum - 1);
}

//TEST FUNCTION
__kernel void distrtest(__global float * lowerbound,
						__global float * delta,
						__global float * presents,
						__global float * pbests,
						unsigned int dimnum,
						unsigned int partnum){

	//get_global_id(1) is dimension number, get_global_id(0) is particle number
	unsigned int i[2]={get_global_id(1), get_global_id(0)*dimnum + get_global_id(1)};

	//does the distribution
	presents[i[1]]=get_global_id(0)*delta[i[0]] + lowerbound[i[0]];
}

//distributes particles linearly between the bounds
__kernel void distribute(__global float * lowerbound,
						 __global float * upperbound,
						 __global float * presents,
						 unsigned int dimnum,
						 unsigned int partnum){
		//get_global_id(1) is dimension number, get_global_id(0) is particle number					 
	uint part=get_global_id(0);	
	uint dim = get_global_id(1);
	uint pdex = part*dimnum + get_global_id(1);

	//distribute the particle between the upper and lower boundaries linearly
	presents[pdex]=part*((upperbound[dim]-lowerbound[dim])/(partnum - 1)) + lowerbound[dim];
}

float rng(uint x, uint y){
	uint z = x+y;
	uint t = z ^ ( z << 11);
	uint out = y ^ (y >> 19) ^ ( t ^ ( t >> 8));
	return (float)(out%1000)/1000;
}

__kernel void update( __global float * presents,
					  __global float * v,
					  float w,
					  unsigned int seed,
					  __global float * pfitnesses,
					  __constant float *upperbound,
					  __global float * pbest,
					  __global float * gbest,
					  __constant float * lowerbound,
					  __global float * fitnesses,
						unsigned int dimnum,
					  float c1,
					  float c2) {
					  
	uint index= get_global_id(0)*dimnum + get_global_id(1);
	uint dex0=get_global_id(0);
	uint dex1=get_global_id(1);

	//get_global_id(0) is partnum, get_global_id(1) is dimiension number
	//velocity update
	v[index]=w*v[index]
	 + c1*rng(seed, index)*(pbest[index]- presents[index])
	 + c2*rng(seed, index)*(gbest[dex1]-presents[index]);

	//position update
	presents[index]=presents[index]+v[index];

	//upper bound check
	if(presents[index]>upperbound[dex1]){
		presents[index]=upperbound[dex1];

	//lower bounds check
	} else if(presents[index]<lowerbound[dex1]){
		presents[index]=lowerbound[dex1];
	}
}

//compares and copies a coordinates into a pbest if necessary
__kernel void update2(__global float * fitnesses,
						unsigned int dimnum,
						__global float * pfitnesses,
						__global float * presents,
						__global float * pbest,
						unsigned int partnum) {

	const unsigned int j=get_global_id(0);
	unsigned int offset=j*dimnum;

	//evaluate fitness of the particle
	/** fitness function is in fitness.cl */
	fitnesses[j]=fitness(presents,offset, dimnum);

	//if the fitness is better than the pfitness, copy the values to pbest array
	if(fitnesses[j]>pfitnesses[j]) {

		//copy new fitness
		pfitnesses[j]=fitnesses[j];

		unsigned int i=dimnum;

		while(i--)
			pbest[offset+i]=presents[offset+i];
	}
}
/*pso.c
The main functions that facilitate running the swarm
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "psocl.h"

#define PREVIOUS_BESTS 5
#define ABS(x) (sqrt((x)*(x)))      //absolute value
#define RAN 2*((float)rand()/RAND_MAX)     //random number between 0 and 1.492 or 0 and 2
#define KERNEL_SIZE (0x100000)

clenv* clinitenv(int dimensionnum, int partnum, float w) {
	int i;
    FILE fPtr=fopen("psocl.cl","r");
    char *kernel_src_str=malloc(sizeof(char)*KERNEL_SIZE);
    if(kernel_src_str==NULL||fPtr==NULL){
        printf("Opening kernel or getting kernel parsing memory failed.\n");
        exit(1);
    }

    clenv * env = malloc(sizeof(clenv));
    if(env==NULL){
        printf("Failed to get memory for cl environment.\n");
        exit(1);
    }

    clGetPlatformIDs(1, &(env->platform_id), &(env->ret_num_platforms));
    clGetDeviceIDs(&env->platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &env->device_id,
	&env->ret_num_devices);
    env->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    env->command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    env->program = clCreateProgramWithSource(env->context, 1, (const char **)&kernel_src_str, KERNEL_SIZE, &env->ret);
	    
    clenv * env=calloc(1,sizeof(clenv));
    if(env==NULL){
        fprintf(stderr,"Failed to allocate memory for the OpenCL environment.\n");
        exit(1);
    }	

    return env;  //return the constructed environment
}

clswarm * initclswarm(clenv * env,int dimensionnum, int particlenum, float weight){
    clswarm * herd=calloc(1,sizeof(clswarm));
    if(herd==NULL){
        fprintf(stderr,"Failed to allocate memory for the OpenCL environment.\n");
        exit(1);
    }
    herd->dimnum=dimensionnum;
    herd->partnum=particlenum;
    herd->iw=weight;

    herd->gbest=clCreateBuffer(env->context, CL_MEM_READ_WRITE, dimensionnum*sizeof(float), NULL, &env->ret);
    herd->pbests=clCreateBuffer(env->context, CL_MEM_READ_WRITE, particlenum*dimensionnum*sizeof(float), NULL, &env->ret);
    herd->bounds=clCreateBuffer(env->context, CL_MEM_READ_WRITE, 2*dimensionnum*sizeof(float), NULL, &env->ret);
    herd->presents=clCreateBuffer(env->context, CL_MEM_READ_WRITE, particlenum*dimensionnum*sizeof(float), NULL, &env->ret);
    herd->pfitnesses=clCreateBuffer(env->context, CL_MEM_READ_WRITE, particlenum*sizeof(float), NULL, &env->ret);
    herd->fitnesses=clCreateBuffer(env->context, CL_MEM_READ_WRITE, particlenum*sizeof(float), NULL, &env->ret);
    herd->vs=clCreateBuffer(env->context, CL_MEM_READ_WRITE, particlenum*dimensionnum*sizeof(float), NULL, &env->ret);

    return herd;
}
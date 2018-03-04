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


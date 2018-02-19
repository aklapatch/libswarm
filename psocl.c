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
#define MAX_SOURCE_SIZE (0x100000)

clswarm* clinitswarm(char type, int dimensionnum, int partnum, float w) {
    int i;
    FILE * fPtr;
    clswarm * school=calloc(1,sizeof(swarm));
    if(school==NULL){
        fprintf(stderr,"Failed to allocate memory for the swarm *\n");
        exit(1);
    }

    cl_platform_id plat_id = NULL;	
	cl_uint ret_num_plats;	
	cl_device_id dev_id = NULL;	
	cl_uint ret_num_devs;	
    cl_ulong local_size;
    cl_int cl_local_size;
    cl_kernel ker;
    cl_event event;

    clGetPlatformIDs(1, &platf_id, &num_plats);
    clGetDeviceIDs(plat_id, CL_DEVICE_TYPE_DEFAULT, 1, &dev_id,	
	&ret_num_devs);
    school->context = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &(school->ret));
    school->command_queue = clCreateCommandQueue(school->context, dev_id, 0, &ret);

    fPtr =fopen("psocl.cl","r");

    char* ker_src_str = malloc(MAX_SOURCE_SIZE*sizeof(char));
    ker_code_size = fread(ker_src_str, 1, MAX_SOURCE_SIZE, fPtr);
    fclose(fPtr);

    school->program=clCreateProgramWithSource(school->context, 1, (const char **)&ker_src_str,	
    (const size_t *)&ker_code_size, &ret);
    clBuildProgram(school->program, 1, &dev_id, "", NULL, NULL);
    ker =clCreateKernel(school->program, "psocl", &ret);


    if(type=='d'||type=='D'){   //for a deep swarm
                            
    }
    else{   //apply swarm properties and its particle array
        school->dimnum=dimensionnum;
        school->partnum=partnum;
        school->w=w;
        school->bounds=(float*)calloc(dimensionnum*2,sizeof(float));
        school->gfitness=-HUGE_VALF;
        school->school=(clparticle*)calloc(partnum,sizeof(clparticle));
        school->gbest=(float*)calloc(dimensionnum,sizeof(float));
        if(school->school==NULL
        ||school->gbest==NULL
        ||school->bounds==NULL){
            fprintf(stderr,"Failed to allocate memory for the array of particles.\n");
            exit(1);
        }

        for(i=0;i<partnum;++i){     //get memory for particle data
            school->school[i].present=(float*)calloc(dimensionnum,sizeof(float));
            school->school[i].v=(float*)calloc(dimensionnum, sizeof(float));
            school->school[i].pbest=(float*)calloc(dimensionnum,sizeof(float));
            school->school[i].pfitness=-HUGE_VALF;
            if(school->school[i].present==NULL
            ||school->school[i].v==NULL
            ||school->school[i].pbest==NULL){
                fprintf(stderr,"Failed to allocate memory for clparticle data.\n");
                exit(1);
            }
        }    
    }

    return school;  //return the constructed swarm
}

//bounds should contain 2X the dimensions in the problem space
//one should be a lower and the other should be an upper bound.
//the program is designed to be intolerant of which is which
//this program also dstributes evenly over the domain.
void cldistributeparticles(clswarm *school,float *bounds){

    int i,j;
    
    for(i=0;i<2*school->dimnum;i+=2){

        //set the bounds for the swarm
        school->bounds[i]=bounds[i];
        school->bounds[i+1]=bounds[i+1];
        float delta=ABS(bounds[i+1]-bounds[i])/(school->partnum-1);

        if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
            for(j=0;j<school->partnum;++j){
                //school.school[j].v[i/2]=3*(RAN-RAN);
                school->school[j].present[i/2]=bounds[i]+school->partnum*delta;
            }
        }
        else{   //if the first bound is higher than the next
            for(j=0;j<school->partnum;++j){
                //school.school[j].v[i/2]=5*(RAN-RAN);
                school->school[j].present[i/2]=bounds[i+1]+school->partnum*delta;
            }
        }
    }
}

void clrunswarm(int iterations, clswarm * school, float (*fitness)(float*)){
    int i,j; 
    srand(time(NULL));

    //the acutal swarm running
    while(iterations--){
        for(i=0;i<school->partnum;++i){
            for(j=0;j<2*school->dimnum;j+=2){

                //velocity update
                school->school[i].v[j/2]=(school->w)*(school->school[i].v[j/2])
                + RAN*(school->school[i].pbest[j/2]- school->school[i].present[j/2])
                + RAN*(school->gbest[j/2]-school->school[i].present[j/2]);

                //position update
                school->school[i].present[j/2]=school->school[i].present[j/2]+school->school[i].v[j/2];
                
                //upper bound check (intolerant of which bound is which)
                if(school->school[i].present[j/2]>((school->bounds[j]>school->bounds[j+1])?school->bounds[j]:school->bounds[j+1])){
                    school->school[i].present[j/2]=(school->bounds[j]>school->bounds[j+1])?school->bounds[j]:school->bounds[j+1];
                }

                //lower bound check (intolerant of which bound is which)
                else if(school->school[i].present[j/2]<((school->bounds[j]<school->bounds[j+1])?school->bounds[j]:school->bounds[j+1])){
                    school->school[i].present[j/2]=(school->bounds[j]<school->bounds[j+1])?school->bounds[j]:school->bounds[j+1];
                }
                
            }
          
            //evaluating how fit the particle is with passed function
            school->school[i].fitness= fitness(school->school[i].present);
            
            //setting particle's best position based on fitness
            if(school->school[i].fitness>school->school[i].pfitness){
                school->school[i].pfitness=school->school[i].fitness;
                memcpy(school->school[i].pbest, school->school[i].present,sizeof(float)*school->dimnum);
            }

            //setting new best particle in the swarm
            //this might be moved inside the previous if statement for optimization
            if(school->school[i].fitness>school->gfitness){
                school->gfitness=school->school[i].fitness;                
                memcpy(school->gbest, school->school[i].present,sizeof(float)*school->dimnum);
            }
        }
    }
}

void clconditionalrunswarm(int iterations, clswarm *school, float (*fitness)(float *), int (*keep_going)(float *)){
    int i,j; 
    srand(time(NULL));

    float ** bests=calloc(PREVIOUS_BESTS,sizeof(float*));

    if(bests==NULL){
        printf("Failed to allocate memory for the array of previous bests.\n");
        releaseswarm(school);
        exit(1);
    }

    for(i=0;i<PREVIOUS_BESTS;++i){
        bests[i]=calloc(school.dimnum,sizeof(float));
            if(bests[i]==NULL){
            printf("Failed to allocate memory for the array of previous bests.\n");
            releaseswarm(school);
            exit(1);
        }
    }

    //the acutal swarm running
    while(iterations--){
        if(keep_going){
            break;
        }

        for(i=0;i<school->partnum;++i){
            for(j=0;j<2*school->dimnum;j+=2){

                //velocity update
                school->school[i].v[j/2]=(school->w)*(school->school[i].v[j/2])
                + RAN*(school->school[i].pbest[j/2]- school->school[i].present[j/2])
                + RAN*(school->gbest[j/2]-school->school[i].present[j/2]);

                //position update
                school->school[i].present[j/2]=school->school[i].present[j/2]+school->school[i].v[j/2];
                
                //upper bound check (intolerant of which bound is which)
                if(school->school[i].present[j/2]>((school->bounds[j]>school->bounds[j+1])?school->bounds[j]:school->bounds[j+1])){
                    school->school[i].present[j/2]=(school->bounds[j]>school->bounds[j+1])?school->bounds[j]:school->bounds[j+1];
                }

                //lower bound check (intolerant of which bound is which)
                else if(school->school[i].present[j/2]<((school->bounds[j]<school->bounds[j+1])?school->bounds[j]:school->bounds[j+1])){
                    school->school[i].present[j/2]=(school->bounds[j]<school->bounds[j+1])?school->bounds[j]:school->bounds[j+1];
                }
                
            }
          
            //evaluating how fit the particle is with passed function
            school->school[i].fitness= fitness(school->school[i].present);
            
            //setting particle's best position based on fitness
            if(school->school[i].fitness>school->school[i].pfitness){
                school->school[i].pfitness=school->school[i].fitness;
                memcpy(school->school[i].pbest, school->school[i].present,sizeof(float)*school->dimnum);
            }

            //setting new best particle in the swarm
            //this might be moved inside the previous if statement for optimization
            if(school->school[i].fitness>school->gfitness){
                school->gfitness=school->school[i].fitness;                
                memcpy(school->gbest, school->school[i].present,sizeof(float)*school->dimnum);
                storebests(school->gbest,bests,school->dimnum);
            }
        }
    }
}

//this stores all the 5 next best bests in an array
void storebests(float*gbest, float **bests,int dimnum){
    int i;
    memcpy(gbest, bests[0],sizeof(float)*dimnum);
    for(i=1;i<PREVIOUS_BESTS;++i){
        memcpy(bests[i-1], bests[i],sizeof(float)*dimnum);
    }
}

//gives the user more explicit access to the best solution
float * clreturnbest(clswarm * school){
    return school->gbest;
}

//frees all data allocated to the swarm
void clreleaseswarm(clswarm * school){
    int i;
    //free particle data
    for(i=0;i<school->partnum;++i){
        free(school->school[i].present);
        free(school->school[i].pbest);
        free(school->school[i].v);
    }
    //free swarm data
    free(school->gbest);
    free(school->school);
    free(school->bounds);

    //free swarm itself
    free(school);
}
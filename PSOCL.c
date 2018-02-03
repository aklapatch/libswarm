/*PSOCL.c
The main functions that facilitate running the swarm
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "PSOCL.h"
#include <time.h>

#define ABS(x) (sqrt((x)*(x)))
#define RAN 1.492*((double)rand()/RAND_MAX)

swarm initswarm(char type, int dimensionnum, int partnum, float w) {
    int i;
    if(type=='d'||type=='D'){   //for a deep swarm
                            
    }
    else{   //apply swarm properties and is particle array
        swarm school;
        school.dimnum=dimensionnum;
        school.partnum=partnum;
        school.w=w;
        school.school=malloc(sizeof(particle)*partnum);
        if(school.school==NULL){
            fprintf(stderr,"Failed to allocate memory for the array of particles.\n");
            exit(1);
        }

        for(i=0;i<partnum;++i){     //get memory for particle data
            school.school[i].present=malloc(sizeof(double)*dimnum);
            school.school[i].v=malloc(sizeof(double)*dimnum);
            school.school[i].pbest=calloc(sizeof(double)*dimnum);
            if(school.school[i].present==NULL
            ||school.school[i].v==NULL
            ||school.school[i].pbest==NULL){
                fprintf(stderr,"Failed to allocate memory for particle data.\n");
                exit(1);
            }
        }    
    }

    return school;  //return the constructed swarm
}

//bounds should contain 2X the dimensions in the problem space
//one should be a lower and the other should be an upper bound.
//the program is designed to be intolerant of which is which
void distributeparticles(swarm school,double *bounds){

    int i,j;

    for(i=0;i<2*school.dimnum;i+=2){
        if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
            int delta=ABS(bounds[i+1]-bounds[i])/school.partnum;
            for (j=0;j<school.partnum;++j){
                school.school[j].present[i/2]=bounds[i]+i*delta/2;
            }
        }
        else{   //if the first bound is higher than the next
            int delta=ABS(bounds[i+1]-bounds[i])/school.partnum;
            for (j=0;j<school.partnum;++j){
                school.school[j].present[i/2]=bounds[i+1]+i*delta/2;
            }
        }
    }
}

void runswarm(int iterations, swarm school, double (*fitness)(double*)){
    int i,j; 
    srand(time(NULL));

    while(iterations--){
        for(i=0;i<school.partnum;++i){
            for(j=0;j<school.dimnum,++j){
                school.school[i].v[j]=school.w*school.school.v[j]
                + RAN*(school.school[i].pbest[j]- school.school[i].present[j])
                + RAN*(school.gbest[j]-school.school[i].present[j]);

                school.school[i].present[j]=school.school[i].present[j]+school.school[i].v[j];
            }

            school.school[i].fitness= fitness(school.school[i].present);
            
            if(school.school[i].fitness>school.school[i].pfitness){
                memcpy(school.school[i].pbest, school.school[i].present,sizeof(double)*school.dimnum);
            }
            if(school.school[i].fitness>school.gfitness){
                memcpy(school.gbest, school.school[i].present,sizeof(double)*school.dimnum);
            }
        }
    }
}

double * returnbest(swarm school){
    double * solution=malloc(sizeof(double)*school.dimnum);
    memcpy(solution,school.gbest,sizeof(double*school.dimnum));
    return solution;
}

void releaseswarm(swarm school){
    int i;
    //free particle data
    for(i=0;i<swarm.partnum;++i){
        free(school.school[i].present);
        free(school.school[i].pbest);
        free(school.school[i].v);
    }
    //free swarm data
    free(school.gbest);
    free(school.school);
}
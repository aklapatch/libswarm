/*psocl.c
The main functions that facilitate running the swarm
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "psocl.h"

#define PREVIOUS_BESTS 5
#define ABS(x) (sqrt((x)*(x)))      //absolute value
#define RAN 2*((double)rand()/RAND_MAX)     //random number between 0 and 1.492 or 0 and 2

swarm* initswarm(char type, int dimensionnum, int partnum, double w) {
    int i;
    swarm * school=calloc(1,sizeof(swarm));
    if(school==NULL){
        fprintf(stderr,"Failed to allocate memory for the swarm *\n");
        exit(1);
    }


    if(type=='d'||type=='D'){   //for a deep swarm
                            
    }
    else{   //apply swarm properties and its particle array
        school->dimnum=dimensionnum;
        school->partnum=partnum;
        school->w=w;
        school->bounds=(double*)calloc(dimensionnum*2,sizeof(double));
        school->gfitness=-HUGE_VALF;
        school->school=(particle*)calloc(partnum,sizeof(particle));
        school->gbest=(double*)calloc(dimensionnum,sizeof(double));
        if(school->school==NULL
        ||school->gbest==NULL
        ||school->bounds==NULL){
            fprintf(stderr,"Failed to allocate memory for the array of particles.\n");
            exit(1);
        }

        for(i=0;i<partnum;++i){     //get memory for particle data
            school->school[i].present=(double*)calloc(dimensionnum,sizeof(double));
            school->school[i].v=(double*)calloc(dimensionnum, sizeof(double));
            school->school[i].pbest=(double*)calloc(dimensionnum,sizeof(double));
            school->school[i].pfitness=-HUGE_VALF;
            if(school->school[i].present==NULL
            ||school->school[i].v==NULL
            ||school->school[i].pbest==NULL){
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
//this program also dstributes evenly over the domain.
void distributeparticles(swarm *school,double *bounds){

    int i,j;
    
    for(i=0;i<2*school->dimnum;i+=2){

        //set the bounds for the swarm
        school->bounds[i]=bounds[i];
        school->bounds[i+1]=bounds[i+1];
        double delta=ABS(bounds[i+1]-bounds[i])/(school->partnum-1);

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

void runswarm(int iterations, swarm * school, double (*fitness)(double*)){
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
                memcpy(school->school[i].pbest, school->school[i].present,sizeof(double)*school->dimnum);
            }

            //setting new best particle in the swarm
            //this might be moved inside the previous if statement for optimization
            if(school->school[i].fitness>school->gfitness){
                school->gfitness=school->school[i].fitness;                
                memcpy(school->gbest, school->school[i].present,sizeof(double)*school->dimnum);
            }
        }
    }
}

void conditionalrunswarm(int iterations, swarm *school, double (*fitness)(double *), int (*keep_going)(double *)){
    int i,j; 
    srand(time(NULL));

    double ** bests=calloc(PREVIOUS_BESTS,sizeof(double*));

    if(bests==NULL){
        printf("Failed to allocate memory for the array of previous bests.\n");
        releaseswarm(school);
        exit(1);
    }

    for(i=0;i<PREVIOUS_BESTS;++i){
        bests[i]=calloc(school.dimnum,sizeof(double));
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
                memcpy(school->school[i].pbest, school->school[i].present,sizeof(double)*school->dimnum);
            }

            //setting new best particle in the swarm
            //this might be moved inside the previous if statement for optimization
            if(school->school[i].fitness>school->gfitness){
                school->gfitness=school->school[i].fitness;                
                memcpy(school->gbest, school->school[i].present,sizeof(double)*school->dimnum);
                storebests(school->gbest,bests,school->dimnum);
            }
        }
    }
}

//this stores all the 5 next best bests in an array
void storebests(double*gbest, double **bests,int dimnum){
    int i;
    memcpy(gbest, bests[0],sizeof(double)*dimnum);
    for(i=1;i<PREVIOUS_BESTS;++i){
        memcpy(bests[i-1], bests[i],sizeof(double)*dimnum);
    }
}

//gives the user more explicit access to the best solution
double * returnbest(swarm * school){
    return school->gbest;
}

//frees all data allocated to the swarm
void releaseswarm(swarm * school){
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
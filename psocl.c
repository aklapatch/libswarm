/*psocl.c
The main functions that facilitate running the swarm
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include "psocl.h"

#define ABS(x) (sqrt((x)*(x)))      //absolute value
#define RAN 1.492*((double)rand()/RAND_MAX)     //random number between 0 and 1.492

swarm initswarm(char type, int dimensionnum, int partnum, double w) {
    int i;
    swarm school;
    if(type=='d'||type=='D'){   //for a deep swarm
                            
    }
    else{   //apply swarm properties and its particle array
        school.dimnum=dimensionnum;
        school.partnum=partnum;
        school.w=w;
        school.bounds=(double*)malloc(dimensionnum*2);
        school.gfitness=-100000000.0;
        school.school=(particle*)malloc(sizeof(particle)*partnum);
        school.gbest=(double*)malloc(sizeof(double)*dimensionnum);
        if(school.school==NULL
        ||school.gbest==NULL
        ||school.bounds==NULL){
            fprintf(stderr,"Failed to allocate memory for the array of particles.\n");
            exit(1);
        }

        for(i=0;i<partnum;++i){     //get memory for particle data
            school.school[i].present=(double*)malloc(sizeof(double)*dimensionnum);
            school.school[i].v=(double*)malloc(sizeof(double)*dimensionnum);
            school.school[i].pbest=(double*)calloc(dimensionnum,sizeof(double));
            school.school[i].pfitness=-100000000.0;
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
//this program also dstributes evenly over the domain.
void distributeparticles(swarm school,double *bounds){

    int i;

    for(i=0;i<2*school.dimnum;i+=2){

        //set the bounds for the swarm
        school.bounds[i]=bounds[i];
        school.bounds[i+1]=bounds[i+1];

        if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
            int delta=ABS(bounds[i+1]-bounds[i])/(2*school.partnum);
            while (school.partnum--){
                school.school[school.partnum].present[i/2]=bounds[i]+i*delta;
            }
        }
        else{   //if the first bound is higher than the next
            int delta=ABS(bounds[i+1]-bounds[i])/(2*school.partnum);
            while (school.partnum--){
                school.school[school.partnum].present[i/2]=bounds[i]+i*delta;
            }
        }
    }
}

void runswarm(int iterations, swarm school, double (*fitness)(double*)){
    int i,j; 
    srand(time(NULL));

    //the acutal swarm running
    while(iterations--){
        for(i=0;i<school.partnum;++i){
            for(j=0;j<2*school.dimnum;j+=2){

                //velocity update
                school.school[i].v[j/2]=(school.w)*(school.school[i].v[j/2])
                + RAN*(school.school[i].pbest[j/2]- school.school[i].present[j/2])
                + RAN*(school.gbest[j/2]-school.school[i].present[j/2]);

                //position update
                school.school[i].present[j/2]=school.school[i].present[j/2]+school.school[i].v[j/2];

                //upper bound check (intolerant of which bound is which)
                if(school.school[i].present[j/2]>((school.bounds[j]>school.bounds[j+1])?school.bounds[j]:school.bounds[j+1])){
                    
                    school.school[i].present[j/2]=(school.bounds[j]>school.bounds[j+1])?school.bounds[j]:school.bounds[j+1];
                }
                //lower bound check (intolerant of which bound is which)
                else if(school.school[i].present[j/2]<(school.bounds[j]<school.bounds[j+1])?school.bounds[j]:school.bounds[j+1]){
                    
                    school.school[i].present[j/2]=(school.bounds[j]<school.bounds[j+1])?school.bounds[j]:school.bounds[j+1];
                }
            }
          
            //evaluating how fit the particle is with passed function
            school.school[i].fitness= fitness(school.school[i].present);
           
            //setting particle's best position based on fitness
            if(school.school[i].fitness>school.school[i].pfitness){
                school.school[i].pfitness=school.school[i].fitness;
                memcpy(school.school[i].pbest, school.school[i].present,sizeof(double)*school.dimnum);
            }

            //setting new best particle in the swarm
            //this might be moved inside the previous if statement for optimization
            if(school.school[i].fitness>school.gfitness){
                school.gfitness=school.school[i].fitness;                
                memcpy(school.gbest, school.school[i].present,sizeof(double)*school.dimnum);
            }
        }
    }
}

//gives the user more explicit access to the best solution
double * returnbest(swarm school){
    return school.gbest;
}

//frees all data allocated to the swarm
void releaseswarm(swarm school){
    int i;
    //free particle data
    for(i=0;i<school.partnum;++i){
        free(school.school[i].present);
        free(school.school[i].pbest);
        free(school.school[i].v);
    }
    //free swarm data
    free(school.gbest);
    free(school.school);
    free(school.bounds);
}

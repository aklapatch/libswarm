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

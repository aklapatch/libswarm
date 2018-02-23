/*psocl.cl
Houses the main function to add the particle positions.
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

typedef struct clparticle {
    float *clpresent, *clpbest, clfitness, clpfitness,*clv;
} particle;

typedef struct clswarm {
    int clpartnum, cldimnum;
    float *clgbest, clgfitness,*clbounds;
    float clw;
    clparticle *clschool;
} clswarm;

#define fitness(x) ((x[0])*(x[0]))
    
int get_global_id(int);

__kernel void distrparts(__global clswarm * school,__global float * bounds){
    int i=2*get_global_id(0);
    int j=get_global_id(1);

    if(i<2*school->dimnum){

        //set the bounds for the swarm
        school->bounds[i]=bounds[i];
        school->bounds[i+1]=bounds[i+1];
        float delta=(bounds[i+1]-bounds[i])/(school->partnum-1);

        if(bounds[i]<bounds[i+1]){  //if the first bound is lower than the next
            if(j<school->partnum){
                //school.school[j].v[i/2]=3*(RAN-RAN);
                school->school[j].present[i]=bounds[i]+school->partnum*delta;
            }
        }
        else{   //if the first bound is higher than the next
            if(j<school->partnum){
                //school.school[j].v[i/2]=5*(RAN-RAN);
                school->school[j].present[i]=bounds[i+1]+school->partnum*delta;
            }
        }
    }
}

__kernel void psoadd( __global clswarm *school, __global float * rand) {
    int i=get_global_id(0);
    int j=2*get_global_id(1);
    int fr=0;

    //velocity update
    school->school[i].v[j/2]=(school->w)*(school->school[i].v[j/2])
     + rand[j]*(school->school[i].pbest[j/2]- school->school[i].present[j/2])
    + rand[j+1]*(school->gbest[j/2]-school->school[i].present[j/2]);

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
          
    //evaluating how fit the particle is with passed function
    school->school[i].fitness= fitness(school->school[i].present);
            
    //setting particle's best position based on fitness
    if(school->school[i].fitness>school->school[i].pfitness){
        school->school[i].pfitness=school->school[i].fitness;
        for(fr=0;fr<school->dimnum;++fr){
            school->school[i].pbest[fr]=school->school[i].present[fr];
        }
    }

    //setting new best particle in the swarm
    //this might be moved inside the previous if statement for optimization
    if(school->school[i].fitness>school->gfitness){
        school->gfitness=school->school[i].fitness;                
        for(fr=0;fr<school->dimnum;++fr){
            school->school[i].pbest[fr]=school->school[i].present[fr];
        }
    }
}



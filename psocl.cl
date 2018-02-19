/*PSOCL.cl
Houses the main function to add the particle positions.
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

typedef struct clparticle {
    double *present, *pbest, fitness, pfitness,*v;
} particle;

typedef struct clswarm {
    int partnum, dimnum;
    double *gbest, gfitness,*bounds;
    float w;
    particle *school;
} clswarm;


int get_global_id(int);

__kernel void distribute_particles(clswarm * school, float * bounds){
    int i=get_global_id(0);
    int j=get_global_id(1);

    if(i<2*school->dimnum){

        //set the bounds for the swarm
        school->bounds[2*i]=bounds[2*i];
        school->bounds[2*i+1]=bounds[2*i+1];
        double delta=ABS(bounds[2*i+1]-bounds[2*i])/(school->partnum-1);

        if(bounds[2*i]<bounds[2*i+1]){  //if the first bound is lower than the next
            if(j<school->partnum){
                //school.school[j].v[i/2]=3*(RAN-RAN);
                school->school[j].present[i]=bounds[2*i]+school->partnum*delta;
            }
        }
        else{   //if the first bound is higher than the next
            if(j<school->partnum){
                //school.school[j].v[i/2]=5*(RAN-RAN);
                school->school[j].present[i]=bounds[2*i+1]+school->partnum*delta;
            }
        }
    }
}

__kernel void distribute_work(clswarm *school){

}

__kernel void vector_add(__global float *rand,
                             __global float *present,
                             __global float *pbest ,
                             __global float *v,
                             __global float *gbest,
                             float w,
                             int  partnum,
                             int dimnum){
                             
    int  dimenid=get_global_id(0);
    
    if((dimenid<dimnum)&&((dimnum+dimenid)<2*dimnum)) {

        //velocity update
        v[dimenid]=w*v[dimenid] + rand[dimenid]*(pbest[dimenid]-present[dimenid]) + rand[dimnum+dimenid]*(gbest[dimenid]-present[dimenid]);

        //present position update
        present[dimenid]=present[dimenid]+v[dimenid];
    }    
}
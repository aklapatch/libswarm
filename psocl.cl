/*PSOCL.cl
Houses the main function to add the particle positions.
Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

int get_global_id(int);

__kernel void PSO_vector_add(__global float *rand,
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
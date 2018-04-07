///return the index with the biggest number
int sort(float * array,int size){
    int ret=0;
    float biggest=-HUGE_VALF;
    while(size-->1){
        if(array[size]>biggest){
            biggest=array[size];
            ret=size;
        }
    }
    return ret;
}

__kernel void compare( __global float *presents,
                        __global float * gbest,
                        __global float * fitnesses,
                        __global float * gfitness,
                        __global int * partnum,
                        __global int * dimnum) {
    
    ///copy most fit particle into the gbest array
    int i,index=sort(fitnesses,*partnum);
    if(fitnesses[index]>*gfitness){
        for(i=0;i<(*dimnum);++i){
            gbest[i]=presents[index*(*partnum)+i];
        }
    }
}
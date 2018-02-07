/*example.c
This is an example, where a swarm is used to find the local max
of a function
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"psocl.h"
#include <math.h>

//example one dimension fitness function
double fitness(double* input) {
    return -(input[0]*input[0]) +27;
}

int main(){
    swarm one = initswarm(' ', 1,10,.01);
    double *answer;
    double bound[2]={-100,100};

    distributeparticles(one,bound);

    runswarm(1000,one,fitness);

    answer=returnbest(one);

    printf("Answer is %f\n",answer[0]);

    releaseswarm(one);
}

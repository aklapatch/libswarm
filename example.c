/*example.c
This is an example, where a swarm is used to find the local max
of a function
*/


#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"PSOCL.h"
#include <math.h>

double fitness(double* input){
    return 1/input[0];
}

int main(){
    swarm one = initswarm(' ', 1,100,1);
    double *answer;
    double bound[2]={1,30};

    distributeparticles(one,bound);

    runswarm(100,one,fitness);

    answer=returnbest(one);

    printf("Answer is %e",answer[0]);

    releaseswarm(one);

}



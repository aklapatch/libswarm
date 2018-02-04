/*tests.c
a couple tests to ensure how C handles array sizes
and random numbers

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define RAN 1.492*((double)rand()/RAND_MAX)
#define SIZE 10

int main(){
    int * testsize= malloc(SIZE*sizeof(int));
    int testarsize[300];
    srand(time(NULL));

    printf("Size of int * %llu.\n",sizeof(testsize));
    printf("Size of the array %llu.\n",sizeof(testarsize));
    //conclusion: array size is printed only for arrays not int *'s
    int i=SIZE;
    //the double typcasting is needed for the number generation to work
    //without it, you just get 0's
    double inf = -HUGE_VALF;
    printf("huge val= %lf\n",inf);
    while(--i)
        printf("Random number rand/rand_max %f.\n",RAN);
}

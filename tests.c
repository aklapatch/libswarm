/*tests.c
a couple tests to ensure how C handles array sizes

Copyright 2018 by Aaron Klapatch,
code derived from http://www.swarmintelligence.org/tutorials.php
along with some help from Dr. Ebeharts presentation at IUPUI.
*/

#include <stdio.h>
#include <stdlib.h>

#define SIZE 300

int main(){
    int * testsize= malloc(SIZE*sizeof(int));
    int testarsize[300];

    printf("Size of int * %d.\n",sizeof(testsize));
    printf("Size of the array %d.\n",sizeof(testarsize));
    //conclusion: size is printed only for arrays not int *'s
}

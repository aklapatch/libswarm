/*example.c
This is an example, where a swarm is used to find the local max
of a function
*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include <math.h>

//example one dimension fitness function
double fitness(double* input) {
    return -(input[0]*input[0]) +27;
}

int main(){

}

///example.cpp
/** a dead simple example of using particle swarm acceleration 
 * Copyright 2018 Aaron Klapatch
 */

#include "oclswarm.hpp"
#include <iostream>
#include <vector>
#include <random>

void printparts(swarm test){

	float ** data= test.getparts();
	float tmp=0;
	cl_uint i=test.getpartnum();
	cl_uint j=test.getdimnum();

	while(i-->1){
		std::cout << "\nParticle " << i << " = " "\n";
		while(j-->1){
			printf(" dimension %f \n", data[i][j]);
			tmp = data[i][j];
		}
		j=test.getdimnum();
	}
}

int main(){
	///the test swarm
	swarm * test=new swarm(1,10,1,.5, 1);



	///make upper and lower bounds and set them
	cl_float lower=-10,  upper=140;
	std::cout << "lower =" << lower << "\n";
	
	///distribute particles
	test->distribute(&lower, &upper);

	printparts(*test);

	int i=10;
	cl_float tmp;
	while(i-->0){
		///run the swarm
		test->update(1);
		//printparts(*test);
		std::cout << "G fitness: " << test->getgfitness() << "\n";
	}

	///get the answer and get it to the user
	cl_float * answer = test->getgbest();
	std::cout<< "The answer is " << answer[0] <<std:: endl;

	std::cout << "gfitness " << test->getgfitness() << std::endl;

	delete test;
	
	return 0;
}
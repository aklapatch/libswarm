///example.cpp
/** a dead simple example of using particle swarm acceleration 
 * Copyright 2018 Aaron Klapatch
 */

#include "clswarm.hpp"
#include <iostream>
#include <vector>
#include <random>

int main(){
	///the test swarm
	clswarm * test=new clswarm(1,10,.3,2, 1.492);

	///make upper and lower bounds and set them
	cl_float lower=-1623,  upper=1674;
	
	///distribute particles
	test->distribute(&lower, &upper);
	cl_float * answer;

	int i=20;
	cl_float tmp;
	while(i-->0){
		///run the swarm
		test->update(1);

		std::cout << "G fitness: " << test->getGFitness() << "\n";

		answer = test->getGBest();
		std::cout<< "The answer is " << answer[0] <<std:: endl;
	}

	///get the answer and get it to the user
	std::cout<< "The answer is " << answer[0] <<std:: endl;

	std::cout << "gfitness " << test->getGFitness() << std::endl;

	delete test;
	
	return 0;
}
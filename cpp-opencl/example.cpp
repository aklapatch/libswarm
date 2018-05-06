///example.cpp
/** a dead simple example of using particle swarm acceleration 
 * Copyright 2018 Aaron Klapatch
 */

#include "oclswarm.hpp"
#include <iostream>
#include <vector>
#include <random>

int main(){
	///the test swarm
	swarm * test=new swarm(1,100,.9);

	///make upper and lower bounds and set them
	cl_float lower=-32,  upper=45;
	
	///distribute particles
	test->distribute(&lower, &upper);
	
	///run the swarm
	test->update(1000);
	
	///get the answer and get it to the user
	float * answer = test->getgbest();
	std::cout<< "The answer is " << answer[0] <<std:: endl;

	std::cout << "gfitness " << test->getgfitness() << std::endl;

	delete test;
	
	return 0;
}
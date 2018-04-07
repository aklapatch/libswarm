///example.cpp
/** a dead simple example of using particle swarm acceleration 
 * Copyright 2018 Aaron Klapatch
 */

#include "oclswarm.hpp"
#include <iostream>
#include <vector>
#include <random>

double fitness(double * in){
	return -(in[0]-2)*(in[0]-2);
}

int main(){
	///the test swarm
	std::cout <<__LINE__ << "\n";
	swarm * test=new swarm(1,100,.9);
	std::cout <<__LINE__ << "\n";

	///make upper and lower bounds and set them
	cl_float lower=-32,  upper=45;
	
	///distribute particles
	test->distribute(&lower, &upper);
	std::cout <<__LINE__ << "\n";
	
	///run the swarm
	test->update(100);
	std::cout <<__LINE__ << "\n";
	
	///get the answer and get it to the user
	float * answer = test->getgbest();
	std::cout<< "The answer is " << answer[0] <<std:: endl;

	delete test;
	
	return 0;
}
	
	
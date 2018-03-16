///example.cpp
/** a dead simple example of using particle swarm acceleration 
 * Copyright 2018 Aaron Klapatch
 */

#include "swarm.hpp"
#include <iostream>
#include <vector>

double fitness(double * in){
	return -(in[0]-2)*(in[0]-2);
}

int main(){
	///the test swarm
	swarm test;

	///make upper and lower bounds and set them
	double lower=-32,  upper=45;
	
	///set the swarm properties
	test.setpartnum(100);
	test.setdimnum(1);
	test.setweight(1);
	
	///distribute particles
	test.distribute(&lower, &upper);
	
	///run the swarm
	test.update(100, fitness);
	
	///get the answer and get it to the user
	double * answer = test.getgbest();
	std::cout<< "The answer is " << answer[0] <<std:: endl;
	
	return 0;
}
	
	
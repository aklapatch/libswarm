///example.cpp
/** a dead simple example of using particle swarm acceleration
 * Copyright 2018 Aaron Klapatch
 */

#include "Swarm.hpp"

double fitness(std::vector<double> in, int offset){
	return -(in[offset]-2)*(in[offset]-2);
}

int main(){
	///the test swarm
	Swarm test(100,1,.1,.5,1);

	///make upper and lower bounds and set them
	double lower=-32,  upper=45;

	///distribute particles
	test.distribute({lower}, {upper});

	///run the swarm
	test.update(100, fitness);

	///get the answer and get it to the user
	std::vector<double> answer = test.getGBest();
	std::cout<< "The answer is " << answer[0] <<std:: endl;

	return 0;
}


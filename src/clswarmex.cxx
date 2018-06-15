//example.cpp
/* a dead simple example of using particle swarm optimization
 * Copyright 2018 Aaron Klapatch
 */

#include "clswarm.hxx"
#include <iostream>

int main(){
	//the test swarm
	clswarm * test=new clswarm(10,1,.2,2, 1.492);

	//make upper and lower bounds and set them
	cl_float lower=-100,  upper=100;
	
	//distribute particles
	test->distribute(&lower, &upper);
	cl_float answer=34;


	cl_float * data= new cl_float[10];
	int j;

	test->getPartData(data);


	for(j=-1;++j<10;)
			std::cout << " i= " << j << " data is " << data[j] << "\n"; 


	int i=20;
	while(i--){
		//run the swarm
		test->update(1);

		std::cout << "G fitness: " << test->getGFitness() << "\n";

		test->getPartData(data);

		std::cout << "particle data\n";
		for(j=-1;++j<10;)
			std::cout << " i= " << j << " data is " << data[j] << "\n"; 

		test->getGBest(&answer);
		std::cout<< "The answer is " << answer <<std:: endl;
	}

	//get the answer and get it to the user
	std::cout<< "The answer is " << answer << std::endl;

	std::cout << "gfitness " << test->getGFitness() << std::endl;

	delete test;
	delete [] data;
	
	return 0;
}
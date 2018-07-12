//example.cpp
/* a dead simple example of using particle swarm optimization
 * Copyright 2018 Aaron Klapatch
 */

#include "clSwarm.hpp"
#include <iostream>

int main(){
	//the test swarm
	clSwarm * test=new clSwarm(10,1,.2,2, 1.492);

	//make upper and lower bounds and set them
	cl_float lower=-1000, upper=1000;

	//distribute particles
	test->distribute(&lower, &upper);
	cl_float answer=34;


	cl_float * data = new cl_float[10];
	int j;

	test->getPartData(data);
	//for(j=-1;++j<10;)
			//std::cout << " i= " << j << " data is " << data[j] << "\n";

	 ///measure time
    auto start = std::chrono::high_resolution_clock::now();
    
	//run the swarm
	test->update(20);

    

    ///std::chrono::duration elapsed = finish - start;



	//std::cout << "Data after 20 passes\n";
	test->getPartData(data);
	test->getGBest(&answer);

	auto finish = std::chrono::high_resolution_clock::now();

	auto msec=std::chrono::duration_cast<std::chrono::milliseconds>(finish-start);

    std::cout <<"Time to execute " << msec.count()<<"\n";

	//for(j=-1;++j<10;)
	//		std::cout << " i= " << j << " data is " << data[j] << "\n";
	
	//std::cout<< "The answer is " << answer <<std:: endl;

	//std::cout << "gfitness " << test->getGFitness() << std::endl;

	delete test;
	delete [] data;

	return 0;
}

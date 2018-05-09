///setsandgets.cpp
/** tests getters and setters for opencl class
 * Copyright 2018 Aaron Klapatch
 */

#include "clswarm.hpp"
#include <iostream>

#define PARTNUM 32
#define DIMNUM 10
#define WEIGHT .3
#define CONST1 1.3
#define CONST2 1.1
#define PDATA 72
#define ERRMARGIN .0001

int main(){

	///make default swarm
	clswarm * test = new clswarm();

	///set particle number
	test->setPartNum(PARTNUM);

	///get particle number and test it
	if (test->getPartNum()!=PARTNUM){
		std::cerr << "Function getPartNum returned an incorrect value.\n";
	}

	//set dimension number
	test->setDimNum(DIMNUM);

	///test result
	if (test->getDimNum()!=DIMNUM){
		std::cerr << "Function getDimNum returned an incorrect value.\n";
	}

	//set weight number
	test->setWeight(WEIGHT);

	///test result
	if (test->getWeight()!=WEIGHT){
		std::cerr << "Function getWeight returned " << test->getWeight()<< " and not " << WEIGHT << "\n";
	}

	//set constants
	test->setConstants(CONST1,CONST2);


	cl_float * ans = test->getConstants();

	///test result
	if (ans[0]!=CONST1||ans[1]!=CONST2){
		std::cerr << "Function getConstants returned " << ans[0] << " and not " << CONST1 << "\n";
		std::cerr << "and " << ans[1] << " and not " << CONST2 << "\n";
	}


	//set particle data
	cl_float * pdata= new cl_float[PARTNUM*DIMNUM];
	int i=PARTNUM*DIMNUM;
	while(i-->0){
		pdata[i]=PDATA;
	}

	///write data to particles
	test->setPartData(pdata);

	//get particle data
	pdata=test->getPartData();

	///test particle data
	i=PARTNUM*DIMNUM;
	bool failed=false;
	while (i-->0){
		if(pdata[i]!=PDATA){
			failed=true;
		}
	}

	///print if it failed
	if(failed==true){
		std::cerr << "Function getPartData returned an incorrect value\n";
	}

	delete [] pdata;

	delete [] ans;
}
__kernel void add(__global float * result, __global int * num,__global float * data){
	*result=0;
	int i;
	for(i=0;i<*num;++i){
		*result+=data[i];
	}
	while(*num){
		*result+=data[*num];
		num[0]=num[0]-1;
	}
}
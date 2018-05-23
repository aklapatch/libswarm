__kernel void add(__global float * result, __global int *num,__global float * data){
	*result=0;
	int i=*num,j=300000;
	///just enought to see the gpu is being used
	while(j--){
		while(i--)
			*result+=data[i];

		i=*num;
	}
}
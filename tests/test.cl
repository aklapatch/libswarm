__kernel void add(__global float * result, __global int *num,__global float * data){
	*result=0;
	int i,j;
	///just enought to see the gpu is being used
	for(j=0;j<4000;++j){
		for(i=0;i<*num;++i){
			*result+=data[i];
		}
	}
	
}
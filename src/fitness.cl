float fitness(__global float * in, int offset,int dimnum){
    return -((in[offset]+27)*(in[offset]+27));
}
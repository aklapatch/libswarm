float fitness(__global float * in, unsigned int offset, unisgned int dimnum){
    return -((in[offset]+27)*(in[offset]+27));
}
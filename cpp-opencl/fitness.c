float fitness(__global float * in, int offset,int dimnum){
    return -(in[offset]*in[offset]);
}
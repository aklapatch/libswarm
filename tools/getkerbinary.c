///getkerbinary.c
/** this is a tool to print out the binary of a kernel source */

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <stdio.h>

int main(int argc, char ** argv){

    //kernel length and kernel data
    unsigned int len=0;
    char * txtin=NULL;

    //cycle through all the kernels
    unsigned int i=argc;
    while(i-->1){
    
        //open file
        FILE * file = fopen(argv[1],"r");

        //find length of file and reset fpointer
        fseek(file,0,SEEK_END);
        len=ftell(file);
        fseek(file,0,SEEK_SET);

        //get memory and read in file
        txtin=(char)malloc((len+1)*sizeof(char));
        fread(txtin,len,file);

        //set final char to \0
        txtin[len]='\0';
        

    }
}

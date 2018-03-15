#include "error.h"

///checks memory allocation
void memcheck(void * Ptr, void (*emergency)()){
    
    ///if the pointer is null, tell user, and run emergency function and exit
    if(Ptr==NULL){
        fprintf(stderr, "Failed to allocate memory.\n");
        emergency();
        exit(1);
    }
}

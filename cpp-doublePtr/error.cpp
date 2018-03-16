#include "error.hpp"

///checks memory allocation
void memcheck(void * Ptr, void (*emergency)()){
    
    ///if the pointer is null, tell user, and run emergency function and exit
    if(Ptr==NULL){
        std::cerr<< "Failed to allocate memory."<<std::endl;
        emergency();
        exit(1);
    }
}

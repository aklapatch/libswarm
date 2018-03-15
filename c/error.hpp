///error.h
/**provides a function to check memory allocation
 */

#ifndef _ERROR_H_
#define _ERROR_H_

#include <iostream>

///checks memory allocation of ptr, and runs function if it is null
void memcheck(void * Ptr, void (*emergency)());

#endif
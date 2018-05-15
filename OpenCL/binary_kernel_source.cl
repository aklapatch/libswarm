///binary_kernel_source.cl
/** includes opencl-c.h to allow clang to compile the kernel */

///include basic opencl functions if clang is compiling the kernel
#ifdef __clang__
#include <opencl-c.h>
#endif

#include "kernels.cl"
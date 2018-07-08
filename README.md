# PSOCL

An OpenCL implementation of particle swarm optimizaion.

Compiled and tested with mingw-w64 v5.0.3 (with mingw32-make) on windows 10.

## Status

Only the CPU version "works" right now.

## Yet-to-do's

1. Get cmake to compile the kernel source files and link them into the library. (I am young in the ways of cmake)

2. Optimize all the library's functions.

3. Figure out if dual command queues is faster or not.

4. Make documentation.

5. Test optimization flags (-O3, -O2, etc.)
cmake_minimum_required (VERSION 3.6)
project (swarm)
#version number
set (swarm_VERSION_MAJOR 0)
set (swarm_VERSION_MINOR 1)

add_library (swarm OpenCL/clswarm.cpp 
                    CPU/swarm.cpp)

install (TARGETS swarm DESTINATION lib)
install(FILES "${PROJECT_BINARY_DIR}/OpenCL/clswarm.hpp" DESTINATION include)
install(FILES "${PROJECT_BINARY_DIR}/CPU/swarm.hpp" DESTINATION include)
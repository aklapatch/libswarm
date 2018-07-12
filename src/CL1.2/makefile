dclex:
	echo R^"( > kernelstring.cl
	type kernels.cl >> kernelstring.cl
	echo )^" >> kernelstring.cl
	clang++ -O2 -Wall clswarmex.cpp clswarm.cpp C:\Windows\System32\OpenCL.dll -o clexample.exe

msvc:
	echo R^"( > kernelstring.cl
	type kernels.cl >> kernelstring.cl
	echo )^" >> kernelstring.cl
	cl /EHsc clSwarmex.cpp clSwarm.cpp /Ox ..\..\OpenCL.lib

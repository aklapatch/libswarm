#tested with mingw-w64 5.0.3 and g++ 7.3.0 (MSYS2) on Windows

example:
	g++ -I. C:\Windows\System32\OpenCL.dll swarmex.cxx swarm.cxx -o example.exe

clexample:
	g++ -I. C:\Windows\System32\OpenCL.dll clswarmex.cxx clswarm.cxx -o clexample.exe

d-example:
	g++ -I. C:\Windows\System32\OpenCL.dll swarmex.cxx swarm.cxx -o d-example.exe

d-clexample:
	g++ -I. clswarmex.cxx clswarm.cxx C:\Windows\System32\OpenCL.dll -o d-clexample.exe

kernel:
	clang -cc1 -cl-std=CL2.0 -cl-single-precision-constant -finclude-default-header kernel-binary.cl

clean:
	del *.exe *.o

uex:
	clang++ -g -Og swarm.cxx swarmex.cxx -o example.out

uexcl:
	g++ -g -Og -lOpenCL clswarm.cxx clswarmex.cxx -o clexample.out
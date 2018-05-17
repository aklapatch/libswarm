#tested with mingw-w64 5.0.3 and g++ 7.3.0 (MSYS2) on Windows

example:
	g++ -I. C:\Windows\System32\OpenCL.dll swarmex.cpp swarm.cpp -o example.exe

clexample:
	g++ -I. C:\Windows\System32\OpenCL.dll clswarmex.cpp clswarm.cpp -o clexample.exe

d-example:
	g++ -I. C:\Windows\System32\OpenCL.dll swarmex.cpp swarm.cpp -o d-example.exe

d-clexample:
	g++ -I. clswarmex.cpp clswarm.cpp C:\Windows\System32\OpenCL.dll -o d-clexample.exe

kernel:
	clang -c -cl-std=CL2.0 -cl-single-precision-constant kernel-binary.cl

clean:
	del *.exe *.o
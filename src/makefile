
clex:
	g++  clswarmex.cpp clswarm.cpp ..\OpenCL.lib -o clexample.exe

dex:
	g++ -g swarmex.cpp swarm.cpp -o dex.exe

dclex:
	g++ -g clswarmex.cpp clswarm.cpp C:\Windows\System32\OpenCL.dll -o dclex.exe

kernel:
	clang -cc1 -cl-std=CL2.0 -cl-single-precision-constant -finclude-default-header kernel-binary.cl

clean:
	del *.exe *.o

uex:
	clang++ -g -Og swarm.cpp swarmex.cpp -o example.out

uexcl:
	clang++ -g -lOpenCL clswarm.cpp clswarmex.cpp -o clexample.out

clex:
	g++  clswarmex.cpp clswarm.cpp ..\OpenCL.lib -o clexample.exe

dclex:
	g++ -Og -g -Wall clswarmex.cpp clswarm.cpp C:\Windows\System32\OpenCL.dll -o clexample.exe

profile:
	gprof clexample.exe gmon.out > analysis.txt

kernel:
	clang -cc1 -cl-std=CL2.0 -finclude-default-header kernels.cl

kertest:
	clang -cc1 -cl-std=CL2.0 -finclude-default-header kernels.cl

clean:
	del *.exe *.o

ex:
	clang++ -g -Og Swarm.cpp swarmex.cpp -o example.out

uexcl:
	clang++ -g -lOpenCL clSwarm.cpp clswarmex.cpp -o clexample.out

profex: ex
	valgrind --tool=callgrind --auto=yes ./example.out

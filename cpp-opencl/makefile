#tested with g++ 7.2.0 mingw-w64 and mingw32-make

example:
	g++ -Wall --std=c++11 -I. C:\Windows\System32\OpenCL.dll example.cpp clswarm.cpp -o example.exe
	
debug:
	g++   --std=c++11 -g -I. C:\Windows\System32\OpenCL.dll example.cpp clswarm.cpp -o d_example.exe

kernel:
	clang -c -cl-std=CL2.0 -cl-single-precision-constant kernels.cl
	
clean:
	del *.exe || rm *.exe

test:
	g++ -Wall -g --std=c++11 -I. C:\Windows\System32\OpenCL.dll testsuite/setsandgets.cpp clswarm.cpp -o setsandgets.exe
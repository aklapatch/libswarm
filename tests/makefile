cpp:
	g++ -Og -g -I. cppwrapper.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe

kernel:
	clang -cc1 -emit-llvm-bc -triple spir-unknown-unknown -cl-std=CL2.0  -finclude-default-header test.cl

#llvm binary is invalid
kertest:
	clang -cc1 -triple spir-unknown-unknown -include opencl-c.h -o test.bc test.cl

cppbin: kertest
	g++ -Og -g -std=c++11 -I. cppbinary.cpp C:\Windows\System32\OpenCL.dll -o cppbin.exe	

clean:
	del *.exe *.o

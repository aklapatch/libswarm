cpp:
	g++ -Og -g -I. cppwrapper.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe

kernel:
	clang -cc1 -emit-llvm-bc -triple spir64-unknown-unknown -cl-std=CL2.0 -include opencl-c.h test.cl
	
cppbin: kernel
	g++ -Og -g -std=c++11 -I. cppbinary.cpp C:\Windows\System32\OpenCL.dll -o cppbin.exe	

clean:
	del *.exe *.o

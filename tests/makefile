cpp:
	g++ -Og -g -I. cppwrapper.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe

kernel:
	clang -c -cl-std=CL2.0 -cl-single-precision-constant test.cl
	
cppbin: kernel
	g++ -Og -g -I. cppbinary.cpp C:\Windows\System32\OpenCL.dll -o cppbin.exe	

clean:
	del *.exe *.o
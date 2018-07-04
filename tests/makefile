unix: usebin.cpp
	g++ -Og -g -lOpenCL usebin.cpp -o usebin 

cpp:
	g++ -Og -g -I. usebin.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe

cppbin: kertest
	g++ -Og -g -std=c++11 -I. cppbinary.cpp C:\Windows\System32\OpenCL.dll -o cppbin.exe	

clean:
	del *.exe *.o

ucppwrap:
	clang++ -Og -g -std=c++11 -lOpenCL cppwrapper.cpp -o cppex.out

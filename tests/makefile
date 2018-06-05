unix: cppbinary.cpp
	g++ -Og -g -lOpenCL cppbinary.cpp -o cppbin 

cpp:
	g++ -Og -g -I. cppwrapper.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe


cppbin: kertest
	g++ -Og -g -std=c++11 -I. cppbinary.cpp C:\Windows\System32\OpenCL.dll -o cppbin.exe	

clean:
	del *.exe *.o

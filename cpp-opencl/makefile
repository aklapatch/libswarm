#tested with g++ 7.2.0 mingw-w64 and mingw32-make

example:
	g++ -Wall --std=c++11 -I. C:\Windows\System32\OpenCL.dll example.cpp oclswarm.cpp -o example.exe
	
debug:
	g++   --std=c++11 -g -I. C:\Windows\System32\OpenCL.dll example.cpp oclswarm.cpp -o d_example.exe
	
clean:
	del *.exe || rm *.exe
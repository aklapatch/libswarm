alloctest:
	gcc -O2 -I. allocspeedtest.c C:\Windows\System32\OpenCL.dll -o alloctest.exe

unixalloctest:
	gcc -O2 -I. allocspeedtest.c -lOpenCL -o alloctest.out

passnum:
	gcc -Og -g -I. clpassnum.c C:\Windows\System32\OpenCL.dll -o passnum.exe

cpp:
	g++ -O2 -I. cppwrapper.cpp C:\Windows\System32\OpenCL.dll -o testcpp.exe

memsize:
	gcc -I. getclmemsize.c C:\Windows\System32\OpenCL.dll -o memsize.exe

clean:
	del *.exe
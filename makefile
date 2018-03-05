CC=clang
CFLAGS=
LINK=ar
LFLAGS= crf

example: example.c pso.c
	$(CC) $(CFLAGS) example.c pso.c -o example.exe
clean:
	del *.exe *.ilk *.pdb *.o *.obj *.a *.dll *.lib
lib: pso.c
	$(CC) $(CFLAGS) -c pso.c
	$(LINK) $(LFLAGS) pso.a pso.o
test: tests.c
	$(CC) $(CFLAGS) tests.c -o test.exe
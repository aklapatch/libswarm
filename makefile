CC=clang
CFLAGS=
LINK=ar
LFLAGS= crf

example: example.c psocl.c
	$(CC) $(CFLAGS) example.c psocl.c -o example.exe
clean:
	del *.exe *.ilk *.pdb *.o *.obj *.a *.dll *.lib
lib: psocl.c
	$(CC) $(CFLAGS) -c psocl.c
	$(LINK) $(LFLAGS) psocl.a PSOCL.o
test: tests.c
	$(CC) $(CFLAGS) tests.c -o test.exe
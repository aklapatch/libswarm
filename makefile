CC=gcc
CFLAGS=
LINK=ar
LFLAGS= crf

example: example.c PSOCL.c
	$(CC) $(CFLAGS) example.c PSOCL.c -o example.exe
clean:
	del *.exe *.ilk *.pdb *.o *.obj *.a *.dll *.lib
lib: PSOCL.c
	$(CC) $(CFLAGS) -c PSOCL.c
	$(LINK) $(LFLAGS) psocl.a PSOCL.o
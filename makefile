CC=clang
CFLAGS= -g

example: example.c PSOCL.c
	$(CC) $(CFLAGS) example.c PSOCL.c -o example.exe
clean:
	del *.exe *.ilk *.pdb *.o *.obj

hw2: main.o pgmUtility.o pgmProcess.o
	nvcc -arch=sm_52 -o hw2 main.o pgmUtility.o pgmProcess.o
main.o: main.c
	gcc -c main.c -o main.o
pgmProcess.o: pgmProcess.cu
	nvcc pgmProcess.cu -c -I.
pgmUtility.o: pgmUtility.c pgmUtility.h
	gcc pgmUtility.c -c -I.
clean:
	rm *.o hw2

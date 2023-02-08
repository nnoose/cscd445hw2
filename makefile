
hw2: main.o pgmUtility.o pgmProcess.o
	nvcc -arch=sm_52 -o hw2 main.o pgmUtility.o pgmProcess.o
main.o: main.cu
	nvcc -c main.cu -o main.o
pgmProcess.o: pgmProcess.cu pgmProcess.h
	nvcc pgmProcess.cu -c -I.
pgmUtility.o: pgmUtility.cu pgmProcess.o
	nvcc pgmUtility.cu pgmProcess.o -c -I.
clean:
	rm *.o hw2

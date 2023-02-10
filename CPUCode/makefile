
hw2CPU:main.o pgmUtility_TianCPU.o
	gcc main.o pgmUtility_TianCPU.o -o hw2CPU -I. -lm

%.o: %.c
	gcc $< -c -I.
clean:
	rm *.o hw2CPU

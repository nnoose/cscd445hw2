
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "pgmProcess.h"
#include "pgmUtility.h"
// Implement or define each function prototypes listed in pgmUtility.h file.
// NOTE: Please follow the instructions stated in the write-up regarding the interface of the functions.
// NOTE: You might have to change the name of this file into pgmUtility.cu if needed.

int * pgmRead( char **header, int *numRows, int *numCols, FILE *in )
{
    int i, j;
    // read in header of the image first
    for( i = 0; i < rowsInHeader; i ++)
    {
        if ( header[i] == NULL )
        {
            return NULL;
        }
        if( fgets( header[i], maxSizeHeadRow, in ) == NULL )
        {
            return NULL;
        }
    }
    // extract rows of pixels and columns of pixels
    sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );  // in pgm the first number is # of cols

    // Now we can intialize the pixel of 2D array, allocating memory
    int *pixels = ( int * ) malloc( ( *numRows ) * ( *numCols ) * sizeof( int ) );

    // read in all pixels into the pixels array.
    for( i = 0; i < *numRows; i ++ )
        for( j = 0; j < *numCols; j ++ )
            if ( fscanf(in, "%d ", &pixels[i*(*numCols)+j]) < 0 )
                return NULL;

    return pixels;
}

int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out )
{
    int i, j;

    // write the header
    for ( i = 0; i < rowsInHeader; i ++ )
    {
        fprintf(out, "%s", *( header + i ) );
    }

    // write the pixels
    for( i = 0; i < numRows; i ++ )
    {
        for ( j = 0; j < numCols; j ++ )
        {
            if ( j < numCols - 1 )
                fprintf(out, "%d ", pixels[i*numCols + j]);
            else
                fprintf(out, "%d\n", pixels[i*numCols+j]);
        }
    }
    return 0;
}



int pgmDrawEdge( int *pixels, int numRows, int numCols, int edgeWidth, char **header )
{

	//init variables. Threads x/y can be defined in header
	clock_t start, end;
	int threadsX = 32;
	int threadsY = 32;
	int blocksX = ceil(numCols/(float)threadsX);
	int blocksY = ceil(numRows/(float)threadsY);
	dim3 grid(blocksX,blocksY,1);
	dim3 block(threadsX,threadsY,1);

	//alloc device array
	int * arr;
	cudaMalloc(&arr,numRows*numCols*sizeof(int));

	start = clock();
	cudaMemcpy(arr,pixels, numRows*numCols*sizeof(int), cudaMemcpyHostToDevice);

	//call kernel
	drawEdge<<<grid,block>>>(arr,numCols,numRows,edgeWidth);

	cudaMemcpy(pixels,arr, numRows*numCols*sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	
	double totalTime = ((double)end-start)/CLOCKS_PER_SEC;
    	printf("Total GPU time taken for Edge: %f\n", totalTime);


	return 0;
}


int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header )
{
	clock_t start, end;
	int threadsX = 32;
	int threadsY = 32;
	int blocksX = ceil(numCols/(float)threadsX);
	int blocksY = ceil(numRows/(float)threadsY);
	dim3 grid(blocksX,blocksY,1);
	dim3 block(threadsX,threadsY,1);
	
	int* d_in =0;
	cudaMalloc(&d_in,numRows*numCols*sizeof(int));
    	int byteSize = numRows*numCols*sizeof(int);
    	
    	start = clock();
    	cudaMemcpy(d_in, pixels, byteSize, cudaMemcpyHostToDevice);
    	drawCircle<<<grid, block>>>(d_in, numCols, numRows, centerCol, centerRow, radius);
    	cudaMemcpy(pixels, d_in, byteSize, cudaMemcpyDeviceToHost);
    	end = clock();
    	
    	double totalTime = ((double)end-start)/CLOCKS_PER_SEC;
    	printf("Total GPU time taken for Circle: %f\n", totalTime);
    
	return 0;
}



int pgmDrawLine( int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    int dx = p2row - p1row, dy = p2col - p1col, steps;
    if (abs(dx) > abs(dy)) steps = abs(dx) + 1;
    else steps = abs(dy) + 1;
    float xInc = (float) (abs(dx) + 1) / (float) steps;
    float yInc = (float) (abs(dy) + 1) / (float) steps;
    if (dx < 0) xInc *= -1;
    if (dy < 0) yInc *= -1;
    int *d_pixels = 0, numBytes = numRows * numCols * sizeof(int);
    cudaMalloc((void **) &d_pixels, numBytes);
    if (d_pixels == 0) {
        printf("Couldn't allocate pixels on device\n");
        return -1;
    }
    cudaMemcpy(d_pixels, pixels, numBytes, cudaMemcpyHostToDevice);
    dim3 grid, block;
    block.x = numCols % 16;
    block.y = numRows % 16;
    grid.x = ceil((float) numRows / block.x);
    grid.y = ceil((float) numCols / block.y);
    drawLine<<<grid, block>>>(d_pixels, steps, xInc, yInc, p1row, p1col, numCols);
    cudaMemcpy(pixels, d_pixels, numBytes, cudaMemcpyDeviceToHost);
    return 0;
}


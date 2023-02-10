
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
	int threadsX = 32;
	int threadsY = 32;
	int blocksX = ceil(numCols/(float)threadsX);
	int blocksY = ceil(numRows/(float)threadsY);
	dim3 grid(blocksX,blocksY,1);
	dim3 block(threadsX,threadsY,1);

	//alloc device array
	int * arr;
	cudaMalloc(&arr,numRows*numCols*sizeof(int));

	cudaMemcpy(arr,pixels, numRows*numCols*sizeof(int), cudaMemcpyHostToDevice);

	//call kernel
	makeEdge<<<grid,block>>>(arr,numCols,numRows,edgeWidth);

	cudaMemcpy(pixels,arr, numRows*numCols*sizeof(int), cudaMemcpyDeviceToHost);


	return 0;
}

int pgmDrawLine( int *pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    int dx = p2row - p1row, dy = p2col - p1col, steps;
    double x = (double) p1row, y = (double) p1col;
    if (abs(dx) > abs(dy)) steps = abs(dx) + 1;
    else steps = abs(dy) + 1;
    float xInc = (float) (abs(dx) + 1) / (float) steps;
    float yInc = (float) (abs(dy) + 1) / (float) steps;
    if (dx < 0) xInc *= -1;
    if (dy < 0) yInc *= -1;
    int indices[steps];
    for (int i = 0; i < steps; i++) {
        int xHalf = 0, yHalf = 0;
        if (fmod(x, 1.0) == .5) {
            x -= .5;
            xHalf = 1;
        }
        if (fmod(y, 1.0) == .5) {
            y -= .5;
            yHalf = 1;
        }
        indices[i] = (int) (round(x) * numCols + round(y));
        if (xHalf) x += .5;
        if (yHalf) y += .5;
        x += xInc;
        y += yInc;
    }
    int *d_pixels = 0, *d_indices = 0, numBytes = numRows * numCols * sizeof(int);
    cudaMalloc((void **) &d_pixels, numBytes);
    if (d_pixels == 0) {
        printf("Couldn't allocate pixels on device\n");
        return -1;
    }
    cudaMemcpy(d_pixels, pixels, numBytes, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_indices, sizeof(indices));
    if (d_indices == 0) {
        printf("Couldn't allocate indices on device\n");
        return -1;
    }
    cudaMemcpy(d_indices, indices, sizeof(indices), cudaMemcpyHostToDevice);
    dim3 grid, block;
    block.x = numCols % 16;
    block.y = numRows % 16;
    grid.x = ceil((float) numRows / block.x);
    grid.y = ceil((float) numCols / block.y);
    pgmDrawLineKernel<<<grid, block>>>(d_pixels, d_indices, numCols, steps);
    cudaMemcpy(pixels, d_pixels, numBytes, cudaMemcpyDeviceToHost);
    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header )
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
    sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );
    int* d_in =0;
    byteSize = sizeof(pixels)/sizeof(int);
    int arraySize = numRows*numCols;
    int gridSize = (int) ceil((float)arraySize/blockSize);
    //cudaMemcpy(d_in, pixels, byteSize, cudaMemcpyHostToDevice);
    //drawCircle<<<gridSize, blockSize>>>(d_in, numCols, numRows, centerCol, centerRow, radius);
    //cudaMemcpy(pixels,d_in, byteSize, cudaMemcpyDeviceToHost);
    
    int x, y;
    int max= 0;
    for(y = 0; y < numRows; y++)
    {
    	for(x = 0; x < numCols; x++)
    	{	
    		int place = (y*numCols)+x;
    		if(pixels[place] > max)
    			max = pixels[place];
    	}
    }
    
    
}

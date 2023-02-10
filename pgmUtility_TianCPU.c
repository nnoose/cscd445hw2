//
//  pgmUtility.c
//  cscd240PGM
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

#include "pgmUtility.h"

//---------------------------------------------------------------------------
int ** pgmRead( char **header, int *numRows, int *numCols, FILE *in )
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
    int **pixels = ( int ** ) malloc( ( *numRows ) * sizeof( int * ) );
    for( i = 0; i < *numRows; i ++)
    {
        pixels[i] = ( int * ) malloc( ( *numCols ) * sizeof( int ) );
        if ( pixels[i] == NULL )
        {
            return NULL;
        }
    }
    
    // read in all pixels into the pixels array.
    for( i = 0; i < *numRows; i ++ )
        for( j = 0; j < *numCols; j ++ )
            if ( fscanf(in, "%d ", *( pixels + i ) + j) < 0 )
                return NULL;
    
    return pixels;
}

//---------------------------------------------------------------------------
//
int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow,
                  int centerCol, int radius, char **header )
{
	int x,y,max;
	for(x = 0; x < numRows; x++)
	{
		for(y = 0; y < numCols; y++)
		{
			int* pixel = {y,x};
			int* center = {centerCol, centerRow};
			int dist = (int) distance( center, pixel);
			int place = (x*numRows)+y;
			if(dist < radius){
				pixels[place] = 0;
			}
			if(pixels[place] > max;){
				max = pixels[place];
			}
		}
	}
	
}

//---------------------------------------------------------------------------
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{

}

//---------------------------------------------------------------------------

int pgmDrawLine( int **pixels, int numRows, int numCols, char **header,
                int p1row, int p1col, int p2row, int p2col )
{

}

//----------------------------------------------------------------------------
int pgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out )
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
                fprintf(out, "%d ", pixels[i][j]);
            else
                fprintf(out, "%d\n", pixels[i][j]);
        }
    }
    return 0;
}

//-------------------------------------------------------------------------------
double distance( int p1[], int p2[] )
{
    return sqrt( pow( p1[0] - p2[0], 2 ) + pow( p1[1] - p2[1], 2 ) );
}



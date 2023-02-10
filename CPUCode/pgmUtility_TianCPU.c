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
	int center[] = {centerCol,centerRow};
	for(int i = 0; i < numRows; i++)
		for(int j = 0; j<numRows;j++)
		{
			int point[] = {j,i};
			double dist = distance(center,point);
			if(dist<=radius)
				pixels[i][j] = 0;
		}
}

//---------------------------------------------------------------------------
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{
	for(int i = 0; i < numRows; i++)
		for(int j = 0; j < numCols; j++)
		{
			if(i<edgeWidth || numRows-i <= edgeWidth || j<edgeWidth || numCols - j <= edgeWidth)
				pixels[i][j] = 0;
		}
}

//---------------------------------------------------------------------------

int pgmDrawLine( int ** pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    int dx = p2row - p1row, dy = p2col - p1col, steps;
    double x = (double) p1row, y = (double) p1col;
    if (abs(dx) > abs(dy)) steps = abs(dx) + 1;
    else steps = abs(dy) + 1;
    float xInc = (float) (abs(dx) + 1) / (float) steps;
    float yInc = (float) (abs(dy) + 1) / (float) steps;
    if (dx < 0) xInc *= -1;
    if (dy < 0) yInc *= -1;
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
        pixels[(int) round(x)][(int) round(y)] = 0;
        if (xHalf) x += .5;
        if (yHalf) y += .5;
        x += xInc;
        y += yInc;
    }
    return 0;
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



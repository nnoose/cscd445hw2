#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pgmUtility.h"

void usage()
{
	printf("Usage:\n-e edgeWidth oldImageFile newImageFile\n");
	printf("-c circleCenterRow circleCenterCol radius oldImageFile newImageFile\n");
	printf("-l p1row p1col p2row p2col oldImageFile newImageFile\n");
}
void freePixels(int ** pixels, int numRows)
{
	for(int i = 0; i < numRows; i++)
		free(pixels[i]);
	free(pixels);
}

void freeHeader(char ** header)
{
	for(int i = 0; i < rowsInHeader; i++)
		free(header[i]);
	free(header);

}

int main(int argc, char ** argv)
{

	//handle bad input cases
	
	if(argc!=5 && argc!= 7 && argc!= 8)
	{
		usage();
		return 1;
	}
	else if(strlen(argv[1])!=2)
	{
		usage();
		return 1;
	}



	//set up variables

	int nRows,nCols;
	int p1y, p1x,p2y,p2x;
	int edgeWidth, circleCenterRow, circleCenterCol, radius;
	clock_t start,end;








	//open the files
	FILE * fin = fopen(argv[argc-2],"r");
	FILE * fout = fopen(argv[argc-1],"w+");


	if(fin==NULL|| fout ==NULL)
	{
		fprintf(stderr, "couldn't open file(s)\n");
		return 1;
	}

	//get what mode we're using
	char mode = argv[1][1];

	//allocate space for the header
	char ** header = (char **)malloc(rowsInHeader*sizeof(char*));
	for(int i = 0; i < rowsInHeader; i++)
		header[i] = (char *)malloc(maxSizeHeadRow);

	//read in the file
	int ** pixels = pgmRead(header, &nRows,&nCols, fin);


	switch(mode){
		case 'c':
			if(argc!=7)
			{
				freePixels(pixels, nRows);
			        freeHeader(header);
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			circleCenterRow = atoi(argv[2]);
        	        circleCenterCol = atoi(argv[3]);
	                radius = atoi(argv[4]);
			start = clock();
			pgmDrawCircle(pixels, nRows, nCols, circleCenterRow, circleCenterCol, radius, header);
			end = clock();
			double timeUsed = ((double) end-start)/CLOCKS_PER_SEC;
			printf("Time of CPU Circle: %f\n",timeUsed);
			break;
		case 'e':
			if(argc!=5)
			{
				freePixels(pixels, nRows);
			        freeHeader(header);
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			edgeWidth = atoi(argv[2]);
			start = clock();
			pgmDrawEdge(pixels, nRows, nCols, edgeWidth, header);
			end = clock();
			double timeUsed = ((double) end-start)/CLOCKS_PER_SEC;
			printf("Time of CPU Edge: %f\n",timeUsed);
			break;
		case 'l':
			if(argc!=8)
			{
				freePixels(pixels, nRows);
			        freeHeader(header);
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			p1y = atoi(argv[2]);
			p1x = atoi(argv[3]);

			p2y = atoi(argv[4]);
			p2x = atoi(argv[5]);
			start = clock();
			pgmDrawLine(pixels,nRows,nCols,header,p1y,p1x,p2y,p2x);
			end = clock();
			double timeUsed = ((double) end-start)/CLOCKS_PER_SEC;
			printf("Time of CPU Line: %f\n",timeUsed);
			

	}

	pgmWrite((const char **)header, (const int **)pixels, nRows, nCols, fout);
	
	freePixels(pixels, nRows);
	freeHeader(header);
	fclose(fin);
	fclose(fout);
	return 0;


}

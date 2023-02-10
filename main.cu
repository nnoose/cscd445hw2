#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

void usage()
{
	printf("Usage:\n-e edgeWidth oldImageFile newImageFile\n");
	printf("-c circleCenterRow circleCenterCol radius oldImageFile newImageFile\n");
	printf("-l p1row p1col p2row p2col oldImageFile newImageFile\n");
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
	int * pixels = pgmRead((char **)header, &nRows,&nCols, fin);



	switch(mode){
		case 'c':
			if(argc!=7)
			{
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			circleCenterRow = atoi(argv[2]);
        	        circleCenterCol = atoi(argv[3]);
	                radius = atoi(argv[4]);


			pgmDrawCircle(pixels, nRows, nCols, circleCenterRow, circleCenterCol, radius, header);
			break;
		case 'e':
			if(argc!=5)
			{
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			edgeWidth = atoi(argv[2]);
			pgmDrawEdge(pixels, nRows, nCols, edgeWidth, header);
			break;
		case 'l':
			if(argc!=8)
			{
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			p1y = atoi(argv[2]);
			p1x = atoi(argv[3]);

			p2y = atoi(argv[4]);
			p2x = atoi(argv[5]);
			//pgmDrawLine(pixels,nRows,nCols,header,p1y,p1x,p2y,p2x);

	}

	pgmWrite((const char **)header, pixels, nRows, nCols, fout);

	for(int i = 0; i < rowsInHeader; i++)
		free(header[i]);
	free(header);

	fclose(fin);
	fclose(fout);
	return 0;


}

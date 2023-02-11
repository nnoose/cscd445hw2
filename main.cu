
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "pgmUtility.h"

void usage()
{
	printf("Usage:\n-e edgeWidth oldImageFile newImageFile\n");
	printf("-c circleCenterRow circleCenterCol radius oldImageFile newImageFile\n");
	printf("-l p1row p1col p2row p2col oldImageFile newImageFile\n");
}

void freeHeader(char ** header)
{
	for(int i = 0; i < rowsInHeader; i++)
		free(header[i]);
	free(header);

}
int isANumber(char * str)
{
	int len = strlen(str);
	for(int i = 0; i < len; i++)
	{
		if(!isdigit(str[i]))
			return 0;
	}
	return 1;
}


int main(int argc, char ** argv)
{

	//handle bad input cases
	if(argc!=5 && argc!= 7 && argc!= 8 && argc != 9)
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
	edgeWidth = 0;

	//open the files
	FILE * fin = fopen(argv[argc-2],"r");
	FILE * fout = fopen(argv[argc-1],"w+");


	if(fin==NULL|| fout ==NULL)
	{
		if(fin != NULL)
			fclose(fin);
		if(fout != NULL)
			fclose(fout);
		usage();
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
			if(argc!=7 && (argc!= 8 || strlen(argv[1])!=3 || argv[1][2]!='e') && (argc != 9 || strlen(argv[2])!=2 || argv[2][1] != 'e'))
			{
				freeHeader(header);
				free(pixels);
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}
			if(argc == 7)
			{
				circleCenterRow = atoi(argv[2]);
       		 	        circleCenterCol = atoi(argv[3]);
		                radius = atoi(argv[4]);

				if(!isANumber(argv[2]) || !isANumber(argv[3]) || !isANumber(argv[4]))
				{
					freeHeader(header);
					free(pixels);
					fclose(fin);
					fclose(fout);
					usage();
					return 1;
				}

			}
			else if(argc == 8)
			{

				circleCenterRow = atoi(argv[2]);
       		 	        circleCenterCol = atoi(argv[3]);
		                radius = atoi(argv[4]);
				edgeWidth  = atoi(argv[5]);

				if(!isANumber(argv[2]) || !isANumber(argv[3]) || !isANumber(argv[4]) || !isANumber(argv[5]))
				{
					freeHeader(header);
					free(pixels);
					fclose(fin);
					fclose(fout);
					usage();
					return 1;
				}
			}
			else if(argc == 9)
			{
				circleCenterRow = atoi(argv[3]);
       		 	        circleCenterCol = atoi(argv[4]);
		                radius = atoi(argv[5]);
				edgeWidth  = atoi(argv[6]);

				if(!isANumber(argv[3]) || !isANumber(argv[4]) || !isANumber(argv[5]) || !isANumber(argv[6]))
				{
					freeHeader(header);
					free(pixels);
					fclose(fin);
					fclose(fout);
					usage();
					return 1;
				}
			}

			pgmDrawCircle(pixels, nRows, nCols, circleCenterRow, circleCenterCol, radius, header);

			if(edgeWidth!= 0)
				pgmDrawEdge(pixels, nRows, nCols, edgeWidth, header);


			break;
		case 'e':
			if(argc!=5 || !isANumber(argv[2]))
			{
				freeHeader(header);
				free(pixels);
				fclose(fin);
				fclose(fout);
				usage();
				return 1;
			}


			edgeWidth = atoi(argv[2]);
			pgmDrawEdge(pixels, nRows, nCols, edgeWidth, header);
			break;
		case 'l':
			if(argc!=8 || !isANumber(argv[2]) || !isANumber(argv[3]) || !isANumber(argv[4]) || !isANumber(argv[5]) )
			{

				free(pixels);
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
			pgmDrawLine(pixels,nRows,nCols,header,p1y,p1x,p2y,p2x);

	}

	pgmWrite((const char **)header, pixels, nRows, nCols, fout);



	free(pixels);
	freeHeader(header);
	fclose(fin);
	fclose(fout);
	return 0;
}



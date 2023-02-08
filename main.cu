#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

int main(int argc, char ** argv)
{

	//set up variables
	//file in
	char fnIn[] = "./balloons.ascii.pgm";

	char fnOut[] = "out.ascii.pgm";
	FILE * fin = fopen(fnIn,"r");
	FILE * fout = fopen(fnOut,"w+");
	int nRows,nCols;


	char ** header = (char **)malloc(rowsInHeader*sizeof(char*));
	for(int i = 0; i < rowsInHeader; i++)
		header[i] = (char *)malloc(maxSizeHeadRow);


	int * pixels = pgmRead((char **)header, &nRows,&nCols, fin);


	//pgmDrawEdge(pixels, nRows, nCols, 50, header);

	pgmWrite((const char **)header, pixels, nRows, nCols, fout);



	for(int i = 0; i < rowsInHeader; i++)
		free(header[i]);
	free(header);

	fclose(fin);
//	fclose(fout);
	return 0;
}

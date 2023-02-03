#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

int main(int argc, char ** argv)
{
	char fnIn[] = "./balloons.ascii.pgm";
	char fnOut[] = "out.ascii.pgm";
	FILE * fin = fopen(fnIn,"r");
	FILE * fout = fopen(fnOut,"w+");
	int nRows,nCols;

	char ** header = malloc(rowsInHeader*sizeof(char*));
	for(int i = 0; i < rowsInHeader; i++)
		header[i] = malloc(maxSizeHeadRow);


	int * pixels = pgmRead((char **)header, &nRows,&nCols, fin);



	pgmWrite((const char **)header, pixels, nRows, nCols, fout);



	for(int i = 0; i < rowsInHeader; i++)
		free(header[i]);
	free(header);

	fclose(fin);
//	fclose(fout);
	return 0;
}

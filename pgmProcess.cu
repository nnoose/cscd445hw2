#include "pgmProcess.h"

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] )
{
	float sqr = (float) (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]);
	return sqrtf(sqr);
}

__global__ void makeEdge(int * arr, int xmax, int ymax, int size)
{
        int x = threadIdx.x+(blockIdx.x*blockDim.x);
        int y = threadIdx.y+(blockIdx.y*blockDim.y);

	if(x<xmax && y < ymax)
	{
		if(x<size||y<size||(ymax-y)<=size||(xmax-x)<=size)
			arr[y*xmax + x] = 0;
	}
}

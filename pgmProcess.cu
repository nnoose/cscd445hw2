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

__global__ void drawCircle(int* pixels, int dimx, int dimy, int centerCol, int centerRow, int radius)
{
	int ix   = blockIdx.x*blockDim.x + threadIdx.x;
    	int iy   = blockIdx.y*blockDim.y + threadIdx.y;
    	int idx = iy*dimx + ix;
    	
    	int center[2] = {centerCol, centerRow};
    	int pixel[2] = {ix,iy};
    	
    	float distance = distance(center, pixel);
    	
    	if(distance <= radius && ix < dimx)
    	{
    		pixels[idx] = 0;
    	}
    	
}
__global__ void drawEdge(int* pixels, int dimx, int dimy, int edgeWidth)

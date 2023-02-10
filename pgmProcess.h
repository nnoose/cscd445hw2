
/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance( int p1[], int p2[] );

__global__ void drawCircle(int* pixels, int dimx, int dimy, int centerCol, int centerRow, int radius);
__global__ void drawEdge(int* pixels, int dimx, int dimy, int edgeWidth);
__global__ void drawLine(int *pixels, int steps, float xInc, float yInc, int p1row, int p1col, int dimx);

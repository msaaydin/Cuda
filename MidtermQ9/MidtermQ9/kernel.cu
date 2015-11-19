#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

__global__ void my_first_kernel(float *x)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	x[tid] = (float)threadIdx.x;
}

int main(int argc, char **argv)
{
	float *h_x, *d_x;
	int   nblocks, nthreads, nsize, n;

	nblocks = 2;
	nthreads = 8;
	nsize = nblocks*nthreads;
	h_x = (float *)malloc(nsize*sizeof(float));
	cudaMalloc((void **)&d_x, nsize*sizeof(float));
	my_first_kernel << <nblocks, nthreads >> >(d_x);

	cudaMemcpy(h_x, d_x, nsize*sizeof(float), cudaMemcpyDeviceToHost);

	for (n = 0; n<nsize; n++)
		printf(" n,  x  =  %d  %f \n", n, h_x[n]);

	cudaFree(d_x);
	free(h_x);
	return 0;
}
// Output:
/*
n, x = 0  0.0
n, x = 1  1.0
n, x = 2  2.0
n, x = 3  3.0
n, x = 4  4.0
n, x = 5  5.0
n, x = 6  6.0
n, x = 7  7.0
n, x = 8  0.0
n, x = 9  1.0
n, x = 10  2.0
n, x = 11  3.0
n, x = 12  4.0
n, x = 13  5.0
n, x = 14  6.0
n, x = 15  7.0


*/
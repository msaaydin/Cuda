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

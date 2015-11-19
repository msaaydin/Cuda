
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define thread_size 128
#include <stdio.h>
#include <math.h>
const long N = 16 * 16;


// CUDA Kernel for Vector Addition
__global__ void Vector_Addition( long *dev_a)
{
	//Get the id of thread within a block	
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < N) // check the boundry condition for the threads
		dev_a[tid] = dev_a[tid] + 45;

}

int main(void)
{
	//Host array
	long Host_a[N];

	//Device array
	long *dev_a;
	long block_size = (int)((N / thread_size) + 0.99);
	//Allocate the memory on the GPU
	cudaMalloc((void **)&dev_a, N*sizeof(long));
	
	//fill the Host array with random elements on the CPU
	for (long i = 0; i <N; i++)
	{
		Host_a[i] = i + 2;
	}
	for (int i = 0; i<100; i++)
		printf(" = %d\n", Host_a[i]);
	printf("************************************************\n");

	//Copy Host array to Device array
	cudaMemcpy(dev_a, Host_a, N*sizeof(long), cudaMemcpyHostToDevice);
	
	//Make a call to GPU kernel
	Vector_Addition <<< block_size, 128 >>> (dev_a);

	//Copy back to Host array from Device array
	cudaMemcpy(Host_a, dev_a, N*sizeof(long), cudaMemcpyDeviceToHost);

	//Display the result
	for (int i = 0; i<100; i++)
		printf(" =%d \n", Host_a[i]);
	//printf("%d + %d = %d",Host_a[400],Host_b[400],Host_c[400]);
	//Free the Device array memory
	cudaFree(dev_a);
	return 0;

}




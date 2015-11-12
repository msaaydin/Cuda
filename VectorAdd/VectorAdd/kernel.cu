
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define thread_size 128;
#include <stdio.h>

const int  N = 400;


// CUDA Kernel for Vector Addition
__global__ void Vector_Addition(const int *dev_a, const int *dev_b, int *dev_c)
{
	//Get the id of thread within a block	
	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < N) // check the boundry condition for the threads
		dev_c[tid] = dev_a[tid] + dev_b[tid];
}


int main(void)
{
		//Host array
		int Host_a[N], Host_b[N], Host_c[N];

		//Device array
		int *dev_a, *dev_b, *dev_c;
		int block_size = N / thread_size;
		//Allocate the memory on the GPU
		cudaMalloc((void **)&dev_a, N*sizeof(int));
		cudaMalloc((void **)&dev_b, N*sizeof(int));
		cudaMalloc((void **)&dev_c, N*sizeof(int));

		//fill the Host array with random elements on the CPU
		for (int i = 0; i <N; i++)
		{
			Host_a[i] = i+2;
			Host_b[i] = i*i;
		}

		//Copy Host array to Device array
		cudaMemcpy(dev_a, Host_a, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, Host_b, N*sizeof(int), cudaMemcpyHostToDevice);

		//Make a call to GPU kernel
		Vector_Addition <<< block_size, 128>>> (dev_a, dev_b, dev_c);

		//Copy back to Host array from Device array
		cudaMemcpy(Host_c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

		//Display the result
		for (int i = 0; i<N; i++)
			printf("%d + %d = %d\n", Host_a[i], Host_b[i], Host_c[i]);

		//Free the Device array memory
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);		
		return 0;

}




#include<stdio.h>
__global__ void add2(int *a)
{
	*a = *a + 285;
}
int main(void)
{
	int *data_h, *data_d;
	cudaMalloc((void**)&data_d, sizeof(int));
	data_h = (int *)malloc(sizeof(int));
	*data_h = 15;
	add2 << <1, 1 >> >(data_d);
	cudaMemcpy(data_h, data_d, sizeof(int),cudaMemcpyDeviceToHost);
	printf("data: %d\n", *data_h);
	free(data_h); cudaFree(data_d);
	return 0;
}

/*
Output :

There is no output:
Explanation: data_d points to
uninitialized memory when add2()
is called.

solution:
//cudaMemcpy( data_d, data_h, sizeof(int) cudaMemcpyHostToDevice );

*/
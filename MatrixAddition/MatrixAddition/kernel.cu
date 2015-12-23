
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

//#define N 512
//#define threadsize 512

#include <stdio.h>
#include <stdlib.h>

#define N 8
#define M 8
#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define B(i,j) B[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout

__global__ void MatAdd(const double *A, const double *B, double *C,
	int rows, int cols)
{
	int row = threadIdx.y;
	int col = threadIdx.x;
	if ((row < rows) && (col < cols)) {
		C(row, col) = A(row, col) + B(row, col);
	}
}

int main(void)
{
	double A[N][M] = { { 1, 2, 3, 4 },
					   { 5, 6, 7, 8 },
					   { 9, 0, 1, 2 } };
	double B[N][M] = { { 3, 3, 3, 3 },
					   { 1, 1, 1, 1 },
	                   { 0, 0, 0, 0 } };
	double *C;
	double *A_d = 0, *B_d = 0, *C_d = 0;
	int rows = N;
	int cols = M;
	dim3 blockDim(M, N);
	C = (double *)malloc(sizeof(*C)*N*M);
	cudaMalloc((void**)&A_d, sizeof(*A_d)*N*M);
	cudaMalloc((void**)&B_d, sizeof(*B_d)*N*M);
	cudaMalloc((void**)&C_d, sizeof(*C_d)*N*M);
	cudaMemcpy(A_d, A, sizeof(*A_d)*N*M, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, sizeof(*B_d)*N*M, cudaMemcpyHostToDevice);
	MatAdd << <1, blockDim >> >(A_d, B_d, C_d, rows, cols);
	cudaMemcpy(C, C_d, sizeof(*C)*N*M, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			printf("%g ", C(i, j));
		}
		printf("\n");
	}
	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(A_d);
	free(C);
	return EXIT_SUCCESS;
}
/*__global__ void matrixAdd(int *a, int *b, int *c) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int index = col + row * N;
	if (col < N && row < N) {

		c[index] = a[index] + b[index];
	}
}


int main() {
	int a[N][N], b[N][N], c[N][N];
	int *dev_a, *dev_b, *dev_c;
	int size = N * N * sizeof(int);
	for (int i = 0; i < N; i++)
	{ 
		for (int j = 0; j < N; j++)
		{
			a[i][j] = i + j;
			b[i][j] = i * j;

		}

	}



	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	int blockSize = size / N;
	
	matrixAdd <<<blockSize,threadsize>>>(dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();

	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			printf("%d ", a[i][j]);
			
		}
		printf("\n");
	}

	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
}*/

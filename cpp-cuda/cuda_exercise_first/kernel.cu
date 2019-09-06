// Note that in this model we do not check
// the error codes and status of kernel call.

/*
In this exercise we will write a simple cuda program that sets the value of an array: A[i] = i. 
Take a look at the file set.cu, that includes a skeleton of the code. 
Here we will complete the code by completing these steps (a TODO exists for each step):

	Allocate memory for the device array d_A
	Free memory for the device array d_A
	Complete the kernel code. The kernel assigns the global thread index to each element in the vector
	Call the kernel with two arguments, pointer to the allocated device memory and the length of the array.
	Copy the result vector from device memory to host memory buffer

Pay close attention to the kernel call parameters, block and grid sizes! 
Can you write the kernel so that it functions even if you launch too many threads?

*/

#include <cstdio>
#include <cmath>

__global__ void set(int* A, int N){
	// TODO 3 - Complete kernel code
	// The kernel assigns the global thread index to each element in the vector

	// 1D grid of 1D blocks
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	A[idx] = idx;
}


int main(void){
	const int N = 128;

	int* d_A;
	int* h_A;

	h_A = (int*)malloc(N * sizeof(int));

	// TODO 1 - Allocate memory for device pointer d_A

	//cudaMalloc(&d_A, N * sizeof(int)));
	cudaMalloc((void**)&d_A, N * sizeof(int));

	// TODO 4 - Call kernel set()
	set<<<2,64>>>(d_A, N);

	// TODO 5 - Copy the results from device memory
	cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%i ", h_A[i]);
	}
	printf("\n");

	free(h_A);

	// TODO 2 - Free memory for device pointer d_A
	//cudaFree(d_A)
	cudaFree((void*)d_A);

	return 0;
}
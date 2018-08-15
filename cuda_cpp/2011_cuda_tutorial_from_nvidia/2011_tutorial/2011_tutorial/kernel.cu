#include <stdio.h>

__global__ void myKernelThatAdds(int *a, int *b, int*c, int n) { 
	/* global keyword is called from host (CPU) and
	is executed on device (GPU).
	We use pointers for the variable inputs
	since the kernel runs on the device, and so the
	variables must point to memory.*/
	
	printf("adding...\n");
	// 0. *c = *a + *b;
	// 1. c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	// 2. c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		c[index] = a[index] + b[index];
	}
	printf("done adding!\n");
}


int main(void) {

	// 1. and 2. length of vector
	// static const int N = 512; 
	// 3. using threads AND blocks
	// 3. together

	static const int N = (2048 * 2048);
	static const int THREADS_PER_BLOCK = 512;

	// 0. instatiate host copies of variables
	// int a, b, c;

	// 1. now instantiate arrays
	int *a, *b, *c;

	// 0. instantiate device copies of variables
	int *d_a, *d_b, *d_c;

	static const int sizeOfInt = sizeof(int);

	// 0. allocate memory space for device copies of a, b, c
	cudaMalloc((void **)&d_a, sizeOfInt);
	cudaMalloc((void **)&d_b, sizeOfInt);
	cudaMalloc((void **)&d_c, sizeOfInt);
	   
	// 0. setup input values
	// a = 2;
	// b = 7;

	// 1. now setup input arrays and alloc space for host copies
	a = (int *)malloc(sizeOfInt);
	b = (int *)malloc(sizeOfInt);
	c = (int *)malloc(sizeOfInt);



	// 0. copy inputs to device copies
	//  cudaMemcpy(d_a, &a, sizeOfInt, cudaMemcpyHostToDevice);
	//  cudaMemcpy(d_b, &b, sizeOfInt, cudaMemcpyHostToDevice);

	cudaMemcpy(d_a, a, sizeOfInt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeOfInt, cudaMemcpyHostToDevice);

	// launch kernel on device
	// myKernelThatAdds <<< 1 , 1 >>> (d_a, d_b, d_c);
	// myKernelThatAdds <<< N , 1 >>> (d_a, d_b, d_c);
	// seems like the blocks*threads should equal
	// the number of things you want to compute
	// in parallel

	// myKernelThatAdds <<< 1 , N >>> (d_a, d_b, d_c);
	myKernelThatAdds <<< (N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (d_a, d_b, d_c, N);


	// copy result back to host
	// cudaMemcpy(&c, d_c, sizeOfInt, cudaMemcpyDeviceToHost);
	cudaMemcpy(c, d_c, sizeOfInt, cudaMemcpyDeviceToHost);


	// clean up memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// now clean up memory copy
	free(a); 	free(b); 	free(c);

	return(0);
}
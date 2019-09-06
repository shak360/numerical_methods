#include <stdio.h>

__global__
void saxpy(int n, float a, float* x, float* y)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

int main(void)
{
	int N = 1 << 20;
	float* x, * y, * d_x, * d_y;
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	// Perform SAXPY on 1M elements
	saxpy <<< (N + 255) / 256, 256 >>> (N, 2.0f, d_x, d_y);
	/*
	In addition to the two calls to the generic host time-stamp function 
	myCPUTimer(), we use the explicit synchronization barrier cudaDeviceSynchronize() 
	to block CPU execution until all previously issued commands on the device have 
	completed. Without this barrier, this code would measure the kernel launch time 
	and not the kernel execution time.
	*/
	//cudaDeviceSynchronize();
	//t2 = myCPUTimer();
	cudaEventRecord(stop);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));
	printf("Max error: %f\n", maxError);

	printf("Max error: %fn", maxError);
	printf("Effective Bandwidth (GB/s): %fn", N * 4 * 3 / milliseconds / 1e6);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}
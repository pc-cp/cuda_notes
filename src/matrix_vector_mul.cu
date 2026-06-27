#include <common_func.h>

#include <gtest/gtest.h>
#include <iostream>
#include <cuda_runtime.h>

// B: [M, M], C[M, 1] --> B x C -> [M, 1]

// each thread process one row for B.
// B is strided access (uncoalesced), C is broadcast
__global__ void matrix_vector_mul(float* A, const float *B, const float *C, int Width) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < Width) {
		float pvalue = 0.0f;
		for (int k = 0; k < Width; ++k) {
			pvalue += B[row * Width + k] * C[k];
		}
		A[row] = pvalue;
	}
}

TEST(matrix_vector_mul, demo0) {
	float *A_h, *B_h, *C_h, *A_golden;
	float *A_d, *B_d, *C_d;

	int Width = 1024;
	int size = Width * sizeof(float);

	CUDA_CHECK(cudaMallocHost((void**)&A_h, size));
	CUDA_CHECK(cudaMallocHost((void**)&B_h, size * Width));
	CUDA_CHECK(cudaMallocHost((void**)&C_h, size));
	CUDA_CHECK(cudaMallocHost((void**)&A_golden, size));

	CUDA_CHECK(cudaMalloc((void**)&A_d, size));
	CUDA_CHECK(cudaMalloc((void**)&B_d, size * Width));
	CUDA_CHECK(cudaMalloc((void**)&C_d, size));
	
	for (int i = 0; i < Width * Width; ++i) {
		B_h[i] = i;
		if (i < Width) {
			C_h[i] = i + 1;
		}
	}
	
	CUDA_CHECK(cudaMemcpy(B_d, B_h, size * Width, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice));

	dim3 grid(ceil(1.0*Width/256), 1, 1);
	dim3 block(256, 1, 1);
	
	matrix_vector_mul<<<grid, block>>>(A_d, B_d, C_d, Width);
	CUDA_CHECK_KERNEL();
	
	CUDA_CHECK(cudaMemcpy(A_h, A_d, size, cudaMemcpyDeviceToHost));

	for (int row = 0; row < Width; ++row) {
		for (int col = 0; col < Width; ++col) {
			A_golden[row] += B_h[row * Width + col] * C_h[col];
		}
	}

	bool is_equal = compare_golden<float>(A_h, A_golden, Width);
	CUDA_CHECK(cudaFree(A_d));
	CUDA_CHECK(cudaFree(B_d));
	CUDA_CHECK(cudaFree(C_d));

	CUDA_CHECK(cudaFreeHost(A_h));
	CUDA_CHECK(cudaFreeHost(B_h));
	CUDA_CHECK(cudaFreeHost(C_h));
	CUDA_CHECK(cudaFreeHost(A_golden));

	std::cout << is_equal << std::endl;
}

#include <common_func.h>
#include <iostream>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// M: j x k
// N: l x k
// P: j x l
// P = M x N^T
// one thread process one element from output array

/**
  this version use strict boundary without corner turning
*/
template <int kTile>
__global__ void matrix_mul_tile_transpose_v1(float* P, float *M, float *N, int j, int k, int l) {
	__shared__ float Ms[kTile][kTile];
	__shared__ float Ns[kTile][kTile];
	
	// this thread process position: [Row, Col] of output
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	int phases = (k + kTile - 1)/kTile;
	float Pvalue = 0.0f;
	for (int phase = 0; phase < phases; ++phase) {
		if (Row >= 0 && Row < j && phase * kTile + threadIdx.x >= 0 && phase * kTile + threadIdx.x < k) {
			Ms[threadIdx.y][threadIdx.x] = M[Row * k + phase * kTile + threadIdx.x];
		}
		else {
			Ms[threadIdx.y][threadIdx.x] = 0.0f;
		}
		if (phase * kTile + threadIdx.y >= 0 && phase * kTile + threadIdx.y < k && Col >= 0 && Col < l) {
			Ns[threadIdx.x][threadIdx.y] = N[Col * k + phase * kTile + threadIdx.y];
		}
		else {
			Ns[threadIdx.x][threadIdx.y] = 0.0f;
		}

		__syncthreads(); // RAW
		for (int t = 0; t < kTile; ++t) {
			Pvalue += Ms[threadIdx.y][t] * Ns[threadIdx.x][t];
		}
		__syncthreads(); // WAR
	}
	if (Row >= 0 && Row < j && Col >= 0 && Col < l) {
		P[Row * l + Col] = Pvalue;
	}
}

/**
  this version use strict boundary with corner turning
*/
template <int kTile>
__global__ void matrix_mul_tile_transpose_v2(float* P, float *M, float *N, int j, int k, int l) {
	__shared__ float Ms[kTile][kTile];
	__shared__ float Ns[kTile][kTile];
	
	// this thread process position: [Row, Col] of output
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	
	int Row_N = blockIdx.x * blockDim.x + threadIdx.y;

	int phases = (k + kTile - 1)/kTile;
	float Pvalue = 0.0f;
	for (int phase = 0; phase < phases; ++phase) {
		if (Row >= 0 && Row < j && phase * kTile + threadIdx.x >= 0 && phase * kTile + threadIdx.x < k) {
			Ms[threadIdx.y][threadIdx.x] = M[Row * k + phase * kTile + threadIdx.x];
		}
		else {
			Ms[threadIdx.y][threadIdx.x] = 0.0f;
		}
		if (phase * kTile + threadIdx.y >= 0 && phase * kTile + threadIdx.y < k && Col >= 0 && Col < l) {
			Ns[threadIdx.y][threadIdx.x] = N[Row_N * k + phase * kTile + threadIdx.x];
		}
		else {
			Ns[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads(); // RAW
		for (int t = 0; t < kTile; ++t) {
			Pvalue += Ms[threadIdx.y][t] * Ns[threadIdx.x][t];
		}
		__syncthreads(); // WAR
	}
	if (Row >= 0 && Row < j && Col >= 0 && Col < l) {
		P[Row * l + Col] = Pvalue;
	}
}

TEST(matrix_mul_tile_transpose, v1) {
	int j = 1024;
	int k = 1024;
	int l = 1024;
	
	float *M_h = nullptr;
	float *N_h = nullptr;
	float *P_h = nullptr;
	float *P_golden = nullptr;
	
	// M: j x k
	// N: l x k
	CUDA_CHECK(cudaMallocHost((void**)&M_h, j * k * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&N_h, k * l * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&P_h, j * l * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&P_golden, j * l * sizeof(float)));

	float *M_d = nullptr;
	float *N_d = nullptr;
	float *P_d = nullptr;

	CUDA_CHECK(cudaMalloc((void**)&M_d, j * k * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&N_d, k * l * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&P_d, j * l * sizeof(float)));

	for (int i = 0; i < j * k; i ++) {
		M_h[i] = i + 1;
	}

	for (int i = 0; i < k * l; ++i) {
		N_h[i] = i + 1;
	}
	
	CUDA_CHECK(cudaMemcpy(M_d, M_h, j * k * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(N_d, N_h, k * l * sizeof(float), cudaMemcpyHostToDevice));
	
	// warm up run
	constexpr int kTile = 8;	
	dim3 grid(ceil(1.0 * j / kTile), ceil(1.0 * l / kTile), 1);
	dim3 block(kTile, kTile, 1);
	matrix_mul_tile_transpose_v1<kTile><<<grid, block>>>(P_d, M_d, N_d, j, k, l);
	CUDA_CHECK_KERNEL();
	
	int iters = 10;
	CudaTimer t;		// records start
	for (int i = 0; i < iters; ++i) {
		matrix_mul_tile_transpose_v1<kTile><<<grid, block>>>(P_d, M_d, N_d, j, k, l);
	}
	float ms = t.elapsed_ms();
	
	// kTile = 16
	printf("matrix_mul_tile_transpose_v1: %.3f ms/iters\n", ms/iters);
	// matrix_mul_tile_transpose_v1: 1.629 ms/iters
	
	// kTile = 32
	// matrix_mul_tile_transpose_v1: 2.858 ms/iters
	
	// kTile = 8;
	// matrix_mul_tile_transpose_v1: 1.265 ms/iters

	CUDA_CHECK(cudaMemcpy(P_h, P_d, j * l * sizeof(float), cudaMemcpyDeviceToHost));
	
	for (int J = 0; J < j; ++J) {
		for (int L = 0; L < l; ++L) {
			float p = 0;
			for (int K = 0; K < k; ++K) {
				p += M_h[J * k + K] * N_h[L * k + K];
			}
			P_golden[J * l + L] = p;
		}
	}

	bool is_equal = compare_golden<float>(P_h, P_golden, j * l);
	std::cout << is_equal << std::endl;
}

TEST(matrix_mul_tile_transpose, v2) {
	int j = 1024;
	int k = 1024;
	int l = 1024;
	
	float *M_h = nullptr;
	float *N_h = nullptr;
	float *P_h = nullptr;
	float *P_golden = nullptr;
	
	// M: j x k
	// N: l x k
	CUDA_CHECK(cudaMallocHost((void**)&M_h, j * k * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&N_h, k * l * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&P_h, j * l * sizeof(float)));
	CUDA_CHECK(cudaMallocHost((void**)&P_golden, j * l * sizeof(float)));

	float *M_d = nullptr;
	float *N_d = nullptr;
	float *P_d = nullptr;

	CUDA_CHECK(cudaMalloc((void**)&M_d, j * k * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&N_d, k * l * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&P_d, j * l * sizeof(float)));

	for (int i = 0; i < j * k; i ++) {
		M_h[i] = i + 1;
	}

	for (int i = 0; i < k * l; ++i) {
		N_h[i] = i + 1;
	}
	
	CUDA_CHECK(cudaMemcpy(M_d, M_h, j * k * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(N_d, N_h, k * l * sizeof(float), cudaMemcpyHostToDevice));
	
	// warm up run
	constexpr int kTile = 8;	
	dim3 grid(ceil(1.0 * j / kTile), ceil(1.0 * l / kTile), 1);
	dim3 block(kTile, kTile, 1);
	matrix_mul_tile_transpose_v2<kTile><<<grid, block>>>(P_d, M_d, N_d, j, k, l);
	CUDA_CHECK_KERNEL();
	
	int iters = 10;
	CudaTimer t;		// records start
	for (int i = 0; i < iters; ++i) {
		matrix_mul_tile_transpose_v2<kTile><<<grid, block>>>(P_d, M_d, N_d, j, k, l);
	}
	float ms = t.elapsed_ms();
	
	// kTile = 16
	printf("matrix_mul_tile_transpose_v2: %.3f ms/iters\n", ms/iters);
	// matrix_mul_tile_transpose_v2: 1.323 ms/iters
	
	// kTile = 32
	// matrix_mul_tile_transpose_v2: 2.386 ms/iters
	
	// kTile = 8;
	// matrix_mul_tile_transpose_v2: 1.132 ms/iters

	CUDA_CHECK(cudaMemcpy(P_h, P_d, j * l * sizeof(float), cudaMemcpyDeviceToHost));
	
	for (int J = 0; J < j; ++J) {
		for (int L = 0; L < l; ++L) {
			float p = 0;
			for (int K = 0; K < k; ++K) {
				p += M_h[J * k + K] * N_h[L * k + K];
			}
			P_golden[J * l + L] = p;
		}
	}

	bool is_equal = compare_golden<float>(P_h, P_golden, j * l);
	std::cout << is_equal << std::endl;
}

#include "common_func.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

// each thread produce one output matrix element
__global__ void matrixMulKernel(float *M, float* N, float *P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < Width) && (col < Width)) {
		float Pvalue = 0.0f;
		for (int k = 0; k < Width; ++k) {
			Pvalue += M[row * Width + k] * N[k * Width + col];
		}
		P[row * Width + col] = Pvalue;
	}
}

/*
 warps execute in lockstep, within a warp, k and col (or row) loop
 variables are identical across all lanes at every instruction. Coalescing is decided purely
 by which array index contains the lane-varying thread index.
 */


// each thread produce one output matrix row
// <<<{1, ceil(1.0*Width/256), 1}, {1, 256, 1}>>>
// pros: ??
// cons: ??
// [viewpoint]: access M is stride Width, N is can coalescing? but M make access N not happen the same time?
// pros: N access is broadcast; simple row-parallel structure.
// cons: strided M access (uncoalesced, main bottleneck); only Width threads total -> severe under-utilization on GPU
// [AI]: M is stried by Width across lanes, which is uncoalesced and is the bottleneck.
//	 N is a uniform broadcast (all lanes same address), which is efficient.
__global__ void matrixMulKernel_v2(float *M, float *N, float *P, int Width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < Width) {
		float Pvalue = 0.0f;
		for (int col = 0; col < Width; ++col) {
			Pvalue = 0.0f;
			for (int k = 0; k < Width; ++k) {
				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
		}
	}
}

// each thread produce one output matrix column.
// <<<{ceil(1.0*Width/256), 1, 1}, {256, 1, 1}>>>
// pros: ??
// cons: ??
// [viewpoint]: access M and N is coalescing. no problem
// [AI]: M is broadcast (same address across lanes),
// 	 N is coalesced (consecutive addresses across lanes) 
// pros: both N (coalesced) and M (broadcast) accesses efficient;
// cons: still only Width threads -> under utilized vs V1;
//	 no data reuse (re-reads M and N from golbal every time - a tiled/SM version)
__global__ void matrixMulKernel_v3(float *M, float *N, float *P, int Width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < Width) {
		float Pvalue = 0.0f;
		for (int row = 0; row < Width; ++row) {
			Pvalue = 0.0f;
			for (int k = 0; k < Width; ++k) {
				Pvalue += M[row * Width + k] * N[k * Width + col];
			}
			P[row * Width + col] = Pvalue;
		}
	}
}

TEST(matrixmul, demo0) {
	float *A_h, *B_h, *C_h, *C_golden;
	float *A_d, *B_d, *C_d;
	int Width = 1024;
	int size = Width * Width * sizeof(float);

	CUDA_CHECK(cudaMallocHost((void**)&A_h, size));
	CUDA_CHECK(cudaMallocHost((void**)&B_h, size));
	CUDA_CHECK(cudaMallocHost((void**)&C_h, size));
	CUDA_CHECK(cudaMallocHost((void**)&C_golden, size));
	
	CUDA_CHECK(cudaMalloc((void**)&A_d, size));
	CUDA_CHECK(cudaMalloc((void**)&B_d, size));
	CUDA_CHECK(cudaMalloc((void**)&C_d, size));
	
	for (int row = 0; row < Width; ++row) {
		for (int col = 0; col < Width; ++col) {
			float pvalue = 0.0f;
			for (int k = 0; k < Width; ++k) {
				pvalue += A_h[row * Width + k] * B_h[k * Width + col];
			}
			C_golden[row * Width + col] = pvalue;
		}
	}

	CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
	
	// warm-up run (first launch includes one-time overhead-discard it)
#define VERSION 2
#if VERSION == 1
	dim3 grids(ceil(1.0 * Width / 16), ceil(1.0 * Width / 16), 1);
	dim3 blocks(16, 16, 1);
	matrixMulKernel<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#elif VERSION == 2
	dim3 grids(1, ceil(1.0 * Width / 256), 1);
	dim3 blocks(1, 256, 1);
	matrixMulKernel_v2<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#elif VERSION == 3
	dim3 grids(ceil(1.0 * Width / 256), 1, 1);
	dim3 blocks(256, 1, 1);
	matrixMulKernel_v3<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#endif
	CUDA_CHECK_KERNEL();

	int iters = 10;
	CudaTimer t;	// records start
	for (int i = 0; i < iters; ++i) {
#if VERSION == 1
	matrixMulKernel<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#elif VERSION == 2
	matrixMulKernel_v2<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#elif VERSION == 3
	matrixMulKernel_v3<<<grids, blocks>>>(A_d, B_d, C_d, Width);
#endif
	}
	float ms = t.elapsed_ms();
#if VERSION == 1
	printf("matmul_v1: %.3f ms/iters\n", ms/iters);
	// matmul_v1: 1.042 ms/iters	
#elif VERSION == 2
	printf("matmul_v2: %.3f ms/iters\n", ms/iters);
	// matmul_v2: 144.511 ms/iters
#elif VERSION == 3
	printf("matmul_v3: %.3f ms/iters\n", ms/iters);
	// matmul_v3: 39.672 ms/iters
#endif

	CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

	for (int row = 0; row < Width; ++row) {
		for (int col = 0; col < Width; ++col) {
			float pvalue = 0.0f;
			for (int k = 0; k < Width; ++k) {
				pvalue += A_h[row * Width + k] * B_h[k * Width + col];
			}
			C_golden[row * Width + col] = pvalue;
		}
	}
	
	bool is_equal = compare_golden<float>(C_h, C_golden, Width * Width);
	
	CUDA_CHECK(cudaFreeHost(A_h));
	CUDA_CHECK(cudaFreeHost(B_h));
	CUDA_CHECK(cudaFreeHost(C_h));
	CUDA_CHECK(cudaFreeHost(C_golden));

	CUDA_CHECK(cudaFree(A_d));
	CUDA_CHECK(cudaFree(B_d));
	CUDA_CHECK(cudaFree(C_d));

	std::cout << is_equal << std::endl;
}

__global__ void hello_kernel() {
	printf("hello from thread %d\n", threadIdx.x);
}

TEST(hello_kernel, demo0) {
	// launch: <<<blocks, threads>>>
	hello_kernel<<<1, 8>>>();

	cudaDeviceSynchronize();
}

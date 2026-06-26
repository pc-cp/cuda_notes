
// each thread produce one output matrix element
__global__ void matrixMulKernel(float *M, float* N,
								float *P, float Width) {
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

// each thread produce one output matrix row
// <<<ceil(1.0*Width/1024), 1024>>>
// pros: ??
// cons: ??
__global__ void matrixMulKernel_v2(float *M, float *N,
								   float *P, float Width) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
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
// <<<ceil(1.0*Width/1024, 1024), 1024>>>
// pros: ??
// cons: ??
__global__ void matrixMulKernel_v3(float *M, float *N,
								   float *P, float Width) {
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



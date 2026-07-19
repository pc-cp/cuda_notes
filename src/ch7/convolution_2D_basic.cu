#include "common_func.h"
#include <cuda/runtime_api.h>
#include <gtest/gtest.h>

__global__ void 
convolution_2D_basic_kernel(float *N, float *F, float *P,
                            int r, int width, int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0.0f;
    if (outRow >= 0 && outRow < height
        && outCol >= 0 && outCol < width) {
        for (int i = -r; i <= r; ++i) {
            for (int j = -r; j <= r; ++j) {
                int row = outRow + i;
                int col = outCol + j;
                if (row >= 0 && row < height
                && col >= 0 && col < width) {
                    tmp += N[row * width + col] * F[(i + r) * (2 * r + 1) + j + r];
                }
                // else {
                //     tmp += 0.0f;
                // }
            }
        }
        P[outRow * width + outCol] = tmp;
    }
}

TEST(convolution_2D_basic_kernel, demo0) {
    float *h_N;
    float *h_F;
    float *h_P;
    float *h_P_golden;

    float *d_N;
    float *d_F;
    float *d_P;

    int r = 2;
    int height = 16;
    int width = 16;

    CUDA_CHECK(cudaMalloc((void**)&d_N, height * width * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_F, (2 * r + 1) * (2 * r + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_P, height * width * sizeof(float)));

    CUDA_CHECK(cudaMallocHost((void**)&h_N, height * width * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_F, (2 * r + 1) * (2 * r + 1) * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_P, height * width * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_P_golden, height * width * sizeof(float)));

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            h_N[i * width + j] = i * j + 2.0f;
        }
    }

    for (int i = 0; i <= 2 * r; ++i) {
        for (int j = 0; j <= 2 * r; ++j) {
            h_F[i * (2 * r + 1) + j] = i * j + 1.0f;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_N, h_N, height * width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F, (2 * r + 1) * (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid {ceil(width/4.0f), ceil(height/4.0f), 1};
    dim3 block{4, 4, 1};

    convolution_2D_basic_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
    
    CUDA_CHECK(cudaMemcpy(h_P, d_P, height * width * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK_KERNEL();

    float temp = 0.0f;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            temp = 0.0f;
            for (i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    int row_ = row + i;
                    int col_ = col + j;
                    if (row_ >= 0 && row_ < height
                    && col_ >= 0 && col_ < width) {
                        temp += h_N[row_ * width + col_] * h_F[(i + r) * (2 * r + 1) + j + r];
                    }
                }
            }
            h_P_golden[row * width + col] = temp;
        }
    }
    
    bool check = compare_golden(h_P, h_P_golden, row * col);
    EXPECT_TRUE(check);

    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFreeHost(h_N));
    CUDA_CHECK(cudaFreeHost(h_F));
    CUDA_CHECK(cudaFreeHost(h_P));
    CUDA_CHECK(cudaFreeHost(h_P_golden));

}
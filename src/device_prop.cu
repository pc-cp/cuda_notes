#include "common_func.h"
#include <iostream>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(device_prop, demo0) {
	int dev_count;
	CUDA_CHECK(cudaGetDeviceCount(&dev_count));
	std::cout << "dev_count: " << dev_count << std::endl;
	
	cudaDeviceProp dev_prop;
	CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, 0));

	std::cout << "ASCII string identifying device: " << dev_prop.name << std::endl;
	std::cout << "Maximum number of resident blocks per multiprocessor: " << dev_prop.maxBlocksPerMultiProcessor << std::endl;
	std::cout << "Maximum resident threads per multiprocessor: " << dev_prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "warpSize: " << dev_prop.warpSize << std::endl;
	std::cout << "max threads per block: " << dev_prop.maxThreadsPerBlock << std::endl;
	
	std::cout << "32-bit registers available per block: " << dev_prop.regsPerBlock << std::endl;
	std::cout << "32-bit registers available per multiprocessor: " << dev_prop.regsPerMultiprocessor << std::endl;

	std::cout << "SMs: " << dev_prop.multiProcessorCount << std::endl;
	std::cout << "the clock frequency (kHz): " << dev_prop.clockRate << std::endl; // HZ?
	std::cout << "compute capability: " << dev_prop.major << "." << dev_prop.minor << std::endl;
// 3080 Ti → 8.6
	// RTX 3080 Ti: CUDA Cores: 10240, Tensor Cores: 320, RT cores: 80;
	// PEak FLOPS: SMs * (cores Per SM) * (clock Hz) * (FLOPS per core per cycle)
	// A fused multiply-add (FMA) counts as 2 FLOPS (one multiply + one add) ancd executes in one cycle. So for peak FP32 with FMA, use 2.
	// 3080Ti: 80 SMs x 128 cores/SM x 1665000000 Hz * 2 FLOPs/core per cycle
	// = 34.1 TFLOPS (FP32)
	
	std::cout << "maximum number of threads allowed along each dim of a block:" << "x: " << dev_prop.maxThreadsDim[0] << ", y: " << dev_prop.maxThreadsDim[1] << ", z: " << dev_prop.maxThreadsDim[2] << std::endl;
	std::cout << "maximum number of blocks allowed along each dim of a grid:" << "x: " << dev_prop.maxGridSize[0] << ", y: " << dev_prop.maxGridSize[1] << ", z: " << dev_prop.maxGridSize[2] << std::endl;
	// std::cout << "Global memory bus width in bits: " << dev_prop.memoryBusWidth << std::endl;
	// std::cout << "Shared memory reserved by CUDA driver per block in bytes: " << dev_prop.reservedSharedMemPerBlock << std::endl;
	std::cout << "Shared memory available per block in bytes: " << dev_prop.sharedMemPerBlock << std::endl;
	std::cout << "Global memory available on device in bytes: " << dev_prop.totalGlobalMem << std::endl;
}

/**
dev_count: 1
ASCII string identifying device: NVIDIA GeForce RTX 3080 Ti
Maximum number of resident blocks per multiprocessor: 16
Maximum resident threads per multiprocessor: 1536
warpSize: 32
max threads per block: 1024
32-bit registers available per block: 65536
32-bit registers available per multiprocessor: 65536
SMs: 80
the clock frequency (kHz): 1665000
compute capability: 8.6
maximum number of threads allowed along each dim of a block:x: 1024, y: 1024, z: 64
maximum number of blocks allowed along each dim of a grid:x: 2147483647, y: 65535, z: 65535
Shared memory available per block in bytes: 49152
Global memory available on device in bytes: 12489195520

 */

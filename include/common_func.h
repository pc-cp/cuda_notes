#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)			\
	do {					\
		cudaError_t err = (call);	\
		if (err != cudaSuccess) {	\
			fprintf(stderr, "CUDA error %s:%d: '%s' returned %d (%s)\n",	\
				__FILE__, __LINE__, #call,				\
				static_cast<int>(err), cudaGetErrorString(err));	\
			exit(EXIT_FAILURE);						\
		}									\
	}while (0)


#define CUDA_CHECK_KERNEL()								\
	do {										\
		CUDA_CHECK(cudaGetLastError()); /* launch config erros*/ 		\
		CUDA_CHECK(cudaDeviceSynchronize()); /* errors during execution*/	\
	} while (0)

template <typename dtype>
bool compare_golden(dtype* result, dtype* golden, int len, float rtol=1e-5f, float atol=1e-5f) {
	int cnt_fail = 0;
	for (int i = 0; i < len; ++i) {
		if (std::isnan(result[i]) || std::isnan(golden[i])) {
			if (!(std::isnan(result[i]) && std::isnan(golden[i]))) {
				printf("%d %d: [nan]: %d vs %d\n", cnt_fail, i, std::isnan(golden[i]), std::isnan(result[i]));
				cnt_fail ++;
			}
		}
		else if (std::fabs(result[i] - golden[i]) > atol + rtol * std::fabs(golden[i])) {
			printf("%d %d: %f vs %f\n", cnt_fail, i, golden[i], result[i]);
			cnt_fail ++;
		}
		if (cnt_fail > 4) break;
	}
	return cnt_fail ? false : true;
}


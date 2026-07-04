#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)			\
	do {					\
		cudaError_t _err = (call);	\
		if (_err != cudaSuccess) {	\
			fprintf(stderr, "CUDA error %s:%d: '%s' returned %d (%s)\n",	\
				__FILE__, __LINE__, #call,				\
				static_cast<int>(_err), cudaGetErrorString(_err));	\
			exit(EXIT_FAILURE);						\
		}									\
	}while (0)


#define CUDA_CHECK_KERNEL()								\
	do {										\
		CUDA_CHECK(cudaGetLastError()); /* launch config erros*/ 		\
		CUDA_CHECK(cudaDeviceSynchronize()); /* errors during execution*/	\
	} while (0)


// below MACRO usage:
/** 
 * cudaEvent_t start, stop;
 * float ms;
 * cudaStream_t stream;
 * CUDA_EVENT_ELAPSED_TIME_BEGIN(start, stop, stream);
 * kernel<<<...>>>();
 * CUDA_EVENT_ELAPSED_TIME_END(start, stop, stream, ms);
 */
#define CUDA_EVENT_ELAPSED_TIME_BEGIN(_start, _stop, _stream)				\
	do {										\
		CUDA_CHECK(cudaEventCreate(&_start));					\
		CUDA_CHECK(cudaEventCreate(&_stop));					\
		CUDA_CHECK(cudaEventRecord(_start, _stream));				\
	} while (0)

#define CUDA_EVENT_ELAPSED_TIME_END(_start, _stop, _stream, _ms)			\
	do {										\
		CUDA_CHECK(cudaEventRecord(_stop, _stream));				\
		CUDA_CHECK(cudaEventSynchronize(_stop));				\
		CUDA_CHECK(cudaEventElapsedTime(&_ms, _start, _stop));			\
		CUDA_CHECK(cudaEventDestroy(_start));					\
		CUDA_CHECK(cudaEventDestroy(_stop));					\
	} while(0)

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


/**
 * CudaTimer t;		// records start
 * kernel<<<>>>();
 * float ms = t.elapsed_ms();	// records stop, syncs, reads
 * std::cout << ms << "ms\n";
 */
class CudaTimer {
	cudaEvent_t start_, stop_;
	cudaStream_t stream_;

	public:
		explicit CudaTimer(cudaStream_t stream = 0): stream_(stream) {
			CUDA_CHECK(cudaEventCreate(&start_));
			CUDA_CHECK(cudaEventCreate(&stop_));
			CUDA_CHECK(cudaEventRecord(start_, stream_));
		}
		
		// no CUDA_CHECK in dtor: must not throw
		~CudaTimer() {
			cudaEventDestroy(start_);
			cudaEventDestroy(stop_);
		}
		
		// call once, after the work
		float elapsed_ms() {
			CUDA_CHECK(cudaEventRecord(stop_, stream_));
			CUDA_CHECK(cudaEventSynchronize(stop_));
			float ms = 0.0f;
			CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
			return ms;
		}
		// events aren't copyable
		CudaTimer(const CudaTimer& timer) = delete;
		CudaTimer& operator=(const CudaTimer& timer) = delete;
};




#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define HANDLE_CUDA_ERROR(statement)                                                                                                                           \
  {                                                                                                                                                            \
    cudaError_t hce_result = (statement);                                                                                                                      \
    if (hce_result != cudaSuccess)                                                                                                                             \
      return hce_result;                                                                                                                                       \
  }

#ifdef __CUDA_ARCH__
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

static constexpr unsigned log2_floor(const unsigned value) {
  unsigned v = value;
  unsigned result = 0;
  while (v >>= 1)
    result++;
  return result;
}

static constexpr unsigned log2_ceiling(const unsigned value) { return value <= 1 ? 0 : log2_floor(value - 1) + 1; }

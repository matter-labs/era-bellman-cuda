#pragma once
#include <cuda_runtime_api.h>

namespace allocator {

template <class T> struct allocation {
  T *ptr;
  cudaStream_t stream;
  allocation();
  operator T *();
  cudaError_t free(cudaStream_t stream);
  cudaError_t free();
  ~allocation();
};

template <class T> cudaError_t allocate(allocation<T> &allocation, size_t size, cudaMemPool_t pool, cudaStream_t stream);

template <class T> cudaError_t free(allocation<T> &allocation, cudaStream_t stream);

} // namespace allocator
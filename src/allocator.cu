#include "allocator.cuh"
#include "common.cuh"
#include "msm_kernels.cuh"

namespace allocator {

template <class T> allocation<T>::allocation() : ptr(nullptr), stream(nullptr){};

template <class T> cudaError_t allocation<T>::free(cudaStream_t new_stream) {
  HANDLE_CUDA_ERROR(cudaFreeAsync(ptr, new_stream));
  ptr = nullptr;
  stream = nullptr;
  return cudaSuccess;
}

template <class T> cudaError_t allocation<T>::free() { return free(stream); }

template <class T> allocation<T>::~allocation() {
  if (ptr != nullptr)
    cudaFreeAsync(ptr, stream);
  ptr = nullptr;
  stream = nullptr;
}

template <class T> allocation<T>::operator T *() { return ptr; }

template struct allocation<ff_storage<8u>>;
template struct allocation<unsigned>;
template struct allocation<void>;
template struct allocation<msm::point_affine>;
template struct allocation<msm::point_projective>;
template struct allocation<msm::point_jacobian>;
template struct allocation<msm::point_xyzz>;

template <class T> size_t get_size_of() { return sizeof(T); }
template <> size_t get_size_of<void>() { return 1; }

template <class T> cudaError_t allocate(allocation<T> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream) {
  T *ptr;
  HANDLE_CUDA_ERROR(cudaMallocFromPoolAsync(&ptr, get_size_of<T>() * size, pool, stream));
  allocation.ptr = ptr;
  allocation.stream = stream;
  return cudaSuccess;
}

template <class T> cudaError_t free(allocation<T> &allocation, cudaStream_t stream) { return allocation.free(stream); }

template cudaError_t allocate<ff_storage<8u>>(allocation<ff_storage<8u>> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<ff_storage<8u>>(allocation<ff_storage<8u>> &allocation, cudaStream_t stream);

template cudaError_t allocate<unsigned>(allocation<unsigned> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<unsigned>(allocation<unsigned> &allocation, cudaStream_t stream);

template cudaError_t allocate<void>(allocation<void> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<void>(allocation<void> &allocation, cudaStream_t stream);

template cudaError_t allocate<msm::point_affine>(allocation<msm::point_affine> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<msm::point_affine>(allocation<msm::point_affine> &allocation, cudaStream_t stream);

template cudaError_t allocate<msm::point_projective>(allocation<msm::point_projective> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<msm::point_projective>(allocation<msm::point_projective> &allocation, cudaStream_t stream);

template cudaError_t allocate<msm::point_jacobian>(allocation<msm::point_jacobian> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<msm::point_jacobian>(allocation<msm::point_jacobian> &allocation, cudaStream_t stream);

template cudaError_t allocate<msm::point_xyzz>(allocation<msm::point_xyzz> &allocation, const size_t size, cudaMemPool_t pool, cudaStream_t stream);
template cudaError_t free<msm::point_xyzz>(allocation<msm::point_xyzz> &allocation, cudaStream_t stream);

} // namespace allocator
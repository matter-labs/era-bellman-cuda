#include "ff_dispatch_st.cuh"
#include "ff_test_kernels.cuh"

template <class FD>
__global__ void fields_add_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.add(x[gid], y[gid]);
}

template <class FD> cudaError_t fields_add(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  fields_add_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_add<fd_q>(const fd_q::storage *, const fd_q::storage *, fd_q::storage *, unsigned);
template <class FD>
__global__ void fields_sub_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.sub(x[gid], y[gid]);
}

template <class FD> cudaError_t fields_sub(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  fields_sub_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_sub<fd_q>(const fd_q::storage *, const fd_q::storage *, fd_q::storage *, unsigned);

template <class FD> __global__ void fields_neg_kernel(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.neg(x[gid]);
}

template <class FD> cudaError_t fields_neg(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  fields_neg_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_neg<fd_q>(const fd_q::storage *, fd_q::storage *, unsigned);

template <class FD> __global__ void fields_to_montgomery_kernel(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.to_montgomery(x[gid]);
}

template <class FD> cudaError_t fields_to_montgomery(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  fields_to_montgomery_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_to_montgomery<fd_q>(const fd_q::storage *, fd_q::storage *, unsigned);

template <class FD> __global__ void fields_from_montgomery_kernel(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.from_montgomery(x[gid]);
}

template <class FD> cudaError_t fields_from_montgomery(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  fields_from_montgomery_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_from_montgomery<fd_q>(const fd_q::storage *, fd_q::storage *, unsigned);

template <class FD>
__global__ void fields_mul_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.mul(x[gid], y[gid]);
}

template <class FD> cudaError_t fields_mul(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  fields_mul_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_mul<fd_q>(const fd_q::storage *, const fd_q::storage *, fd_q::storage *, unsigned);

template <class FD> __global__ void fields_dbl_kernel(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const FD fd = FD();
  result[gid] = fd.dbl(x[gid]);
}

template <class FD> cudaError_t fields_dbl(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  fields_dbl_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(x, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t fields_dbl<fd_q>(const fd_q::storage *, fd_q::storage *, unsigned);

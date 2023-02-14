#include "ec_test_kernels.cuh"

template <class POINT1, class POINT2 = POINT1> __global__ void ec_add_kernel(const POINT1 *x, const POINT2 *y, POINT1 *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  fd_p fd = fd_p();
  result[gid] = curve::add(x[gid], y[gid], fd);
}

template <class POINT1, class POINT2> cudaError_t ec_add(const POINT1 *x, const POINT2 *y, POINT1 *result, const unsigned count) {
  ec_add_kernel<POINT1, POINT2><<<count / 32, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t ec_add<pp, pa>(const pp *, const pa *, pp *, unsigned);
template cudaError_t ec_add<pp>(const pp *, const pp *, pp *, unsigned);
template cudaError_t ec_add<pz, pa>(const pz *, const pa *, pz *, unsigned);
template cudaError_t ec_add<pz>(const pz *, const pz *, pz *, unsigned);

template <class POINT1, class POINT2 = POINT1> __global__ void ec_sub_kernel(const POINT1 *x, const POINT2 *y, POINT1 *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  fd_p fd = fd_p();
  result[gid] = curve::sub(x[gid], y[gid], fd);
}

template <class POINT, class POINT2> cudaError_t ec_sub(const POINT *x, const POINT2 *y, POINT *result, const unsigned count) {
  ec_sub_kernel<POINT, POINT2><<<count / 32, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t ec_sub<pp, pa>(const pp *, const pa *, pp *, unsigned);
template cudaError_t ec_sub<pp>(const pp *, const pp *, pp *, unsigned);
template cudaError_t ec_sub<pz, pa>(const pz *, const pa *, pz *, unsigned);
template cudaError_t ec_sub<pz>(const pz *, const pz *, pz *, unsigned);

template <class POINT> __global__ void ec_dbl_kernel(const POINT *x, POINT *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  fd_p fd = fd_p();
  result[gid] = curve::dbl(x[gid], fd);
}

template <class POINT> cudaError_t ec_dbl(const POINT *x, POINT *result, const unsigned count) {
  ec_dbl_kernel<<<count / 32, 32>>>(x, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t ec_dbl<pp>(const pp *, pp *, unsigned);
template cudaError_t ec_dbl<pz>(const pz *, pz *, unsigned);

template <class POINT> __global__ void ec_mul_kernel(const fd_q::storage *x, const POINT *y, POINT *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  fd_p fd = fd_p();
  result[gid] = curve::mul<fd_q>(x[gid], y[gid], fd);
}

template <class POINT> cudaError_t ec_mul(const fd_q::storage *x, const POINT *y, POINT *result, const unsigned count) {
  ec_mul_kernel<<<count / 32, 32>>>(x, y, result, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

template cudaError_t ec_mul<pp>(const fd_q::storage *, const pp *, pp *, unsigned);
template cudaError_t ec_mul<pz>(const fd_q::storage *, const pz *, pz *, unsigned);

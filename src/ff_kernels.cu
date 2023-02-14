#include "common.cuh"
#include "ff_dispatch_st.cuh"
#include "ff_kernels.cuh"
#include "memory.cuh"
#include <array>
#include <cassert>

namespace ff {

__constant__ powers_data powers_data_w;
__constant__ powers_data powers_data_g_f;
__constant__ powers_data powers_data_g_i;

template <typename FD> __device__ __forceinline__ typename FD::storage get_power_of_w(const unsigned index, bool inverse) {
  return get_power<FD>(powers_data_w, index, inverse);
}

template <typename FD> __device__ __forceinline__ typename FD::storage get_power_of_g(const unsigned index, bool inverse) {
  return inverse ? get_power<FD>(powers_data_g_i, index, false) : get_power<FD>(powers_data_g_f, index, false);
}

__global__ void precompute_powers_kernel(const fd_q::storage value, fd_q::storage *powers, const unsigned offset, const unsigned count) {
  typedef typename fd_q::storage storage;
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  storage result = fd_q::CONFIG::one;
  storage base = value;
  for (unsigned e = gid << offset; e != 0; e >>= 1, base = fd_q::sqr(base))
    if (e & 1)
      result = fd_q::mul(result, base);
  powers[gid] = result;
}

cudaError precompute_powers(const fd_q::storage value, fd_q::storage *powers, const unsigned offset, const unsigned count) {
  const unsigned threads_per_block = 32;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  precompute_powers_kernel<<<grid_dim, block_dim>>>(value, powers, offset, count);
  return cudaGetLastError();
}

cudaError_t set_up_powers(const fd_q::storage value, const powers_data &symbol, const unsigned coarse_log_count) {
  typedef typename fd_q::storage storage;
  const unsigned fine_log_count = fd_q::CONFIG::omega_log_order - coarse_log_count;
  const unsigned fine_count = 1 << fine_log_count;
  const unsigned coarse_count = 1 << coarse_log_count;
  const unsigned fine_mask = fine_count - 1;
  const unsigned coarse_mask = coarse_count - 1;
  powers_data data{fine_log_count, coarse_log_count, fine_mask, coarse_mask};
  HANDLE_CUDA_ERROR(cudaMalloc(&data.fine, sizeof(storage) << fine_log_count));
  HANDLE_CUDA_ERROR(cudaMalloc(&data.coarse, sizeof(storage) << coarse_log_count));
  HANDLE_CUDA_ERROR(precompute_powers(value, data.fine, 0, fine_count));
  HANDLE_CUDA_ERROR(precompute_powers(value, data.coarse, fine_log_count, coarse_count));
  HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(symbol, &data, sizeof(powers_data)));
  return cudaSuccess;
}

cudaError_t get_powers_data(const powers_data &symbol, powers_data &data) { return cudaMemcpyFromSymbol(&data, symbol, sizeof(powers_data)); }

cudaError_t tear_down_powers(const powers_data &symbol) {
  powers_data data{};
  HANDLE_CUDA_ERROR(get_powers_data(symbol, data));
  HANDLE_CUDA_ERROR(cudaFree(data.fine));
  HANDLE_CUDA_ERROR(cudaFree(data.coarse));
  return cudaSuccess;
}

cudaError_t set_up_powers_of_w(const unsigned coarse_log_count) { return set_up_powers(fd_q::CONFIG::omega, powers_data_w, coarse_log_count); }

cudaError_t tear_down_powers_of_w() { return tear_down_powers(powers_data_w); }

cudaError_t set_up_powers_of_g_f(const unsigned coarse_log_count) {
  const auto value = fd_q::to_montgomery({fd_q::CONFIG::omega_generator});
  return set_up_powers(value, powers_data_g_f, coarse_log_count);
}

cudaError_t set_up_powers_of_g_i(const unsigned coarse_log_count) {
  const auto value = fd_q::inverse(fd_q::to_montgomery({fd_q::CONFIG::omega_generator}));
  return set_up_powers(value, powers_data_g_i, coarse_log_count);
}

cudaError_t tear_down_powers_of_g_f() { return tear_down_powers(powers_data_g_f); }

cudaError_t tear_down_powers_of_g_i() { return tear_down_powers(powers_data_g_i); }

cudaError_t get_powers_data_w(powers_data &data) { return get_powers_data(powers_data_w, data); }

cudaError_t get_powers_data_g_f(powers_data &data) { return get_powers_data(powers_data_g_f, data); }

cudaError_t get_powers_data_g_i(powers_data &data) { return get_powers_data(powers_data_g_i, data); }

template <class STORAGE> __global__ void set_value_by_ref_kernel(STORAGE *target, const STORAGE *value, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto v = memory::load<STORAGE, memory::ld_modifier::cg>(value);
  memory::store<STORAGE, memory::st_modifier::cs>(target + gid, v);
}

template <class STORAGE> cudaError_t set_value_by_ref(STORAGE *target, const STORAGE *value, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  set_value_by_ref_kernel<STORAGE><<<grid_dim, block_dim, 0, stream>>>(target, value, count);
  return cudaGetLastError();
}

template cudaError_t set_value_by_ref<ff_storage<8>>(ff_storage<8> *target, const ff_storage<8> *value, unsigned count, cudaStream_t stream);

template <class STORAGE> __global__ void set_value_by_val_kernel(STORAGE *target, const STORAGE value, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  memory::store<STORAGE, memory::st_modifier::cs>(target + gid, value);
}

template <class STORAGE> cudaError_t set_value_by_val(STORAGE *target, const STORAGE *value, const unsigned count, cudaStream_t stream) {
  STORAGE value_val{};
  memcpy(&value_val, value, sizeof(STORAGE));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  set_value_by_val_kernel<STORAGE><<<grid_dim, block_dim, 0, stream>>>(target, value_val, count);
  return cudaGetLastError();
}

template cudaError_t set_value_by_val<ff_storage<8>>(ff_storage<8> *target, const ff_storage<8> *value, unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::cg>(x + gid);
  auto rs = FD::mul(as, xs);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_by_ref(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::cg>(x + gid);
  auto rs = FD::mul(a, xs);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_by_val(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void a_plus_x_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::cg>(x + gid);
  auto rs = FD::add(as, xs);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t a_plus_x_by_ref(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count,
                            cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  a_plus_x_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, result, count);
  return cudaGetLastError();
}

template cudaError_t a_plus_x_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void a_plus_x_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::cg>(x + gid);
  auto rs = FD::add(a, xs);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t a_plus_x_by_val(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count,
                            cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  a_plus_x_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, result, count);
  return cudaGetLastError();
}

template cudaError_t a_plus_x_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void x_plus_y_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::add(xs, ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t x_plus_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  x_plus_y_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t x_plus_y<fd_q>(const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_plus_y_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y,
                                        typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::add(FD::mul(as, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_plus_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                             const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_plus_y_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_plus_y_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count,
                                            cudaStream_t stream);

template <class FD>
__global__ void ax_plus_y_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, const typename FD::storage *y,
                                        typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::add(FD::mul(a, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_plus_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                             const unsigned count, cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_plus_y_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_plus_y_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count,
                                            cudaStream_t stream);

template <class FD>
__global__ void x_minus_y_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::sub(xs, ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t x_minus_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  x_minus_y_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t x_minus_y<fd_q>(const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_minus_y_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y,
                                         typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::sub(FD::mul(as, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_minus_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_minus_y_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_minus_y_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result,
                                             const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_minus_y_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, const typename FD::storage *y,
                                         typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::sub(FD::mul(a, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_minus_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              const unsigned count, cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_minus_y_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_minus_y_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result,
                                             const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void x_minus_ay_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y,
                                         typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::sub(xs, FD::mul(as, ys));
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t x_minus_ay_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  x_minus_ay_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t x_minus_ay_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result,
                                             const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void x_minus_ay_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, const typename FD::storage *y,
                                         typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::sub(xs, FD::mul(a, ys));
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t x_minus_ay_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              const unsigned count, cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  x_minus_ay_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t x_minus_ay_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result,
                                             const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void x_mul_y_kernel(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::mul(xs, ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t x_mul_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  x_mul_y_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t x_mul_y<fd_q>(const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void ax_mul_y_by_ref_kernel(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y,
                                       typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto as = memory::load<storage, memory::ld_modifier::ca>(a);
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::mul(FD::mul(as, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_mul_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                            const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_mul_y_by_ref_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_mul_y_by_ref<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count,
                                           cudaStream_t stream);

template <class FD>
__global__ void ax_mul_y_by_val_kernel(const typename FD::storage a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                                       const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::ca>(x + gid);
  auto ys = memory::load<storage, memory::ld_modifier::ca>(y + gid);
  auto rs = FD::mul(FD::mul(a, xs), ys);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t ax_mul_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                            const unsigned count, cudaStream_t stream) {
  typedef typename FD::storage storage;
  storage a_val{};
  memcpy(&a_val, a, sizeof(storage));
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  ax_mul_y_by_val_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(a_val, x, y, result, count);
  return cudaGetLastError();
}

template cudaError_t ax_mul_y_by_val<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, const unsigned count,
                                           cudaStream_t stream);

template <class FD> __global__ void naive_inverse_kernel(const typename FD::storage *x, typename FD::storage *result, const unsigned count) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto xs = memory::load<storage, memory::ld_modifier::cg>(x + gid);
  auto rs = FD::inverse(xs);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD> cudaError_t naive_inverse(const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  naive_inverse_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(x, result, count);
  return cudaGetLastError();
}

template cudaError_t naive_inverse<fd_q>(const fd_q::storage *x, fd_q::storage *result, const unsigned count, cudaStream_t stream);

template <class FD>
__global__ void get_powers_of_w_kernel(typename FD::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                                       const bool bit_reverse) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned shift = FD::CONFIG::omega_log_order - log_degree;
  const unsigned raw_index = gid + offset;
  const unsigned index = bit_reverse ? __brev(raw_index) >> (32 - log_degree) : raw_index;
  const unsigned shifted_index = index << shift;
  auto rs = get_power_of_w<FD>(shifted_index, inverse);
  memory::store<storage, memory::st_modifier::cs>(target + gid, rs);
}

template <class FD>
cudaError_t get_powers_of_w(typename FD::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                            const bool bit_reverse, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  get_powers_of_w_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(target, log_degree, offset, count, inverse, bit_reverse);
  return cudaGetLastError();
}

template cudaError_t get_powers_of_w<fd_q>(fd_q::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                                           const bool bit_reverse, cudaStream_t stream);

template <class FD>
__global__ void get_powers_of_g_kernel(typename FD::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                                       const bool bit_reverse) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned raw_index = gid + offset;
  const unsigned index = bit_reverse ? __brev(raw_index) >> (32 - log_degree) : raw_index;
  auto rs = get_power_of_g<FD>(index, inverse);
  memory::store<storage, memory::st_modifier::cs>(target + gid, rs);
}

template <class FD>
cudaError_t get_powers_of_g(typename FD::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                            const bool bit_reverse, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  get_powers_of_g_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(target, log_degree, offset, count, inverse, bit_reverse);
  return cudaGetLastError();
}

template cudaError_t get_powers_of_g<fd_q>(fd_q::storage *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse,
                                           const bool bit_reverse, cudaStream_t stream);

template <class FD>
__global__ void omega_shift_kernel(const typename FD::storage *values, typename FD::storage *result, const unsigned log_degree, const unsigned shift,
                                   const unsigned offset, const unsigned count, const bool inverse) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  auto value = memory::load<storage, memory::ld_modifier::cg>(values + gid);
  const unsigned degree_shift = FD::CONFIG::omega_log_order - log_degree;
  const unsigned index = shift * (gid + offset) << degree_shift;
  auto pow = get_power_of_w<FD>(index, inverse);
  auto rs = FD::mul(value, pow);
  memory::store<storage, memory::st_modifier::cs>(result + gid, rs);
}

template <class FD>
cudaError_t omega_shift(const typename FD::storage *values, typename FD::storage *result, const unsigned log_degree, const unsigned shift,
                        const unsigned offset, const unsigned count, const bool inverse, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  omega_shift_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(values, result, log_degree, shift, offset, count, inverse);
  return cudaGetLastError();
}

template cudaError_t omega_shift<fd_q>(const fd_q::storage *values, fd_q::storage *result, const unsigned log_degree, const unsigned shift,
                                       const unsigned offset, const unsigned count, const bool inverse, cudaStream_t stream);

template <class FD>
__global__ void batch_inverse_per_thread_kernel(const typename FD::storage *inputs, typename FD::storage *scratch, typename FD::storage *outputs,
                                                const unsigned count) {
  // If the kernel is acting in-place, outputs aliases inputs. If not, outputs may alias scratch.
  typedef typename FD::storage storage;

  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  const int grid_size = int(blockDim.x * gridDim.x);
  int i = int(gid);

  // If count < grid size, the kernel is inefficient no matter what (because each thread processes just one element)
  // but we should still bail out if a thread has no assigned elems at all.
  storage running_prod = FD::get_one();
  for (; i < count; i += grid_size) {
    memory::store<storage, memory::st_modifier::cg>(scratch + i, running_prod);
    running_prod = FD::template mul(running_prod, memory::load<storage, memory::ld_modifier::cg>(inputs + i));
  }

  auto inv = FD::inverse(running_prod);

  i -= grid_size;
  for (; i >= 0; i -= grid_size) {
    const auto input = memory::load<storage, memory::ld_modifier::cs>(inputs + i);
    const auto fwd_scan = memory::load<storage, memory::ld_modifier::cs>(scratch + i);
    // Isolates and stores this input's inv
    memory::store<storage, memory::st_modifier::cs>(outputs + i, FD::template mul(fwd_scan, inv));
    // Removes this input's inv contribution
    if (i - grid_size >= 0)
      inv = FD::template mul(inv, input);
  }
}

template <class FD>
cudaError_t batch_inverse_per_thread(const typename FD::storage *inputs, typename FD::storage *scratch, typename FD::storage *outputs, const unsigned count,
                                     cudaStream_t stream) {
  const unsigned block_dim = 256;
  int device;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
  // useful for heuristic tuning in case we need it
  int max_blocks_per_sm;
  HANDLE_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, batch_inverse_per_thread_kernel<FD>, block_dim, 0));
  // 256 threads per block * 3 blocks per SM is a reasonable choice on A100, but there's a broad plateau of values in the
  // (block dim x blocks per SM) plane that yield similar performance for n=2^25 and n=2^26, so I'm not sure (3, 256) is the absolute best.
  int blocks_per_sm = 3;
  assert(blocks_per_sm <= max_blocks_per_sm);
  auto wave = blocks_per_sm * prop.multiProcessorCount;
  auto blocks_based_on_count = (count + block_dim - 1) / block_dim;
  const dim3 grid_dim = (blocks_based_on_count < wave) ? blocks_based_on_count : wave;
  batch_inverse_per_thread_kernel<FD><<<grid_dim, block_dim, 0, stream>>>(inputs, scratch, outputs, count);
  return cudaGetLastError();
}

template cudaError_t batch_inverse_per_thread<fd_q>(const fd_q::storage *inputs, fd_q::storage *scratch, fd_q::storage *outputs, const unsigned count,
                                                    cudaStream_t stream);

template <class STORAGE, unsigned LOG_SPLIT = 0>
__global__ void bit_reverse_in_place_kernel(const std::array<STORAGE *, 1 << LOG_SPLIT> values, const unsigned log_count, const unsigned partition_id = 0) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned threads_count = 1 << (log_count - LOG_SPLIT);
  if (gid >= threads_count)
    return;
  const unsigned l_index = gid + threads_count * partition_id;
  const unsigned r_index = __brev(l_index) >> (32 - log_count);
  if (l_index >= r_index)
    return;
  const auto l_values = values[l_index / threads_count] + (l_index % threads_count);
  const auto r_values = values[r_index / threads_count] + (r_index % threads_count);
  const auto l_value = memory::load<STORAGE, memory::ld_modifier::cg>(l_values);
  const auto r_value = memory::load<STORAGE, memory::ld_modifier::cg>(r_values);
  memory::store<STORAGE, memory::st_modifier::cs>(l_values, r_value);
  memory::store<STORAGE, memory::st_modifier::cs>(r_values, l_value);
}

template <class STORAGE, unsigned LOG_SPLIT = 0>
__global__ void bit_reverse_kernel(const std::array<const STORAGE *, 1 << LOG_SPLIT> values, const std::array<STORAGE *, 1 << LOG_SPLIT> results,
                                   const unsigned log_count, const unsigned partition_id = 0) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned threads_count = 1 << (log_count - LOG_SPLIT);
  if (gid >= threads_count)
    return;
  const unsigned i_index = gid + threads_count * partition_id;
  const unsigned o_index = __brev(i_index) >> (32 - log_count);
  const auto i_values = values[i_index / threads_count] + (i_index % threads_count);
  const auto o_values = results[o_index / threads_count] + (o_index % threads_count);
  const auto value = memory::load<STORAGE, memory::ld_modifier::cg>(i_values);
  memory::store<STORAGE, memory::st_modifier::cs>(o_values, value);
}

template <class STORAGE, unsigned LOG_SPLIT>
cudaError_t bit_reverse_single(const STORAGE **values, STORAGE **results, const unsigned log_count, const unsigned partition_id, cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const unsigned threads_count = 1 << (log_count - LOG_SPLIT);
  const dim3 block_dim = threads_count < threads_per_block ? threads_count : threads_per_block;
  const dim3 grid_dim = (threads_count - 1) / block_dim.x + 1;
  auto v = *reinterpret_cast<std::array<const STORAGE *, 1 << LOG_SPLIT> *>(values);
  auto r = *reinterpret_cast<std::array<STORAGE *, 1 << LOG_SPLIT> *>(results);
  if (values[0] == results[0])
    bit_reverse_in_place_kernel<STORAGE, LOG_SPLIT><<<grid_dim, block_dim, 0, stream>>>(r, log_count, partition_id);
  else
    bit_reverse_kernel<STORAGE, LOG_SPLIT><<<grid_dim, block_dim, 0, stream>>>(v, r, log_count, partition_id);
  return cudaGetLastError();
}

template <class STORAGE>
cudaError_t bit_reverse_single(const STORAGE **values, STORAGE **results, const unsigned log_count, const unsigned partition_id, cudaStream_t stream,
                               const unsigned log_split) {
  switch (log_split) {
  case 0:
    return bit_reverse_single<STORAGE, 0>(values, results, log_count, partition_id, stream);
  case 1:
    return bit_reverse_single<STORAGE, 1>(values, results, log_count, partition_id, stream);
  case 2:
    return bit_reverse_single<STORAGE, 2>(values, results, log_count, partition_id, stream);
  default:
    assert(log_split < 3);
    return cudaSuccess;
  }
}

template cudaError_t bit_reverse_single(const fd_q::storage **values, fd_q::storage **results, const unsigned log_count, const unsigned partition_id,
                                        cudaStream_t stream, const unsigned log_split);

template <class STORAGE> __global__ void select_kernel(const STORAGE *source, STORAGE *destination, const unsigned *indexes, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned index = indexes[gid];
  const auto value = memory::load<STORAGE, memory::ld_modifier::cg>(source + index);
  memory::store<STORAGE, memory::st_modifier::cs>(destination + gid, value);
}

template <class STORAGE> cudaError_t select(const STORAGE *source, STORAGE *destination, const unsigned *indexes, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 96;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  select_kernel<STORAGE><<<grid_dim, block_dim, 0, stream>>>(source, destination, indexes, count);
  return cudaGetLastError();
}

template cudaError_t select<fd_q::storage>(const fd_q::storage *source, fd_q::storage *destination, const unsigned *indexes, const unsigned count,
                                           cudaStream_t stream);
} // namespace ff

#pragma once

#include "ff_dispatch_st.cuh"
#include "ff_kernels.cuh"

namespace ff {

cudaError_t set_up(unsigned powers_of_w_coarse_log_count, unsigned powers_of_g_coarse_log_count);

template <class STORAGE> cudaError_t set_value(STORAGE *target, const STORAGE *value, unsigned count, cudaStream_t stream);

template <class STORAGE> cudaError_t set_value_zero(STORAGE *target, unsigned count, cudaStream_t stream);

template <class FD> cudaError_t set_value_one(typename FD::storage *target, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t a_plus_x(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_plus_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count,
                      cudaStream_t stream);

template <class FD>
cudaError_t ax_minus_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                       unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_minus_ay(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                       unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_mul_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count,
                     cudaStream_t stream);

struct grand_product_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  fd_q::storage *inputs;
  fd_q::storage *outputs;
  unsigned count;
};

cudaError_t grand_product(const grand_product_configuration &configuration);

cudaError_t grand_product_reverse(const grand_product_configuration &configuration);

struct multiply_by_powers_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  fd_q::storage *values;
  fd_q::storage *base;
  fd_q::storage *result;
  unsigned count;
};

cudaError_t multiply_by_powers(const multiply_by_powers_configuration &configuration);

struct inverse_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  fd_q::storage *inputs;
  fd_q::storage *outputs;
  unsigned count;
};

cudaError_t inverse(const inverse_configuration &configuration);

struct poly_evaluate_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  fd_q::storage *values;
  fd_q::storage *point;
  fd_q::storage *result;
  unsigned count;
};

cudaError_t poly_evaluate(const poly_evaluate_configuration &configuration);

cudaError_t get_powers_of_w(unsigned log_degree, unsigned offset, unsigned count, bool inverse);

cudaError_t get_powers_of_g(unsigned offset, unsigned count, bool inverse);

template <class STORAGE> cudaError_t bit_reverse(const STORAGE *values, STORAGE *result, unsigned log_count, cudaStream_t stream);

template <class STORAGE>
cudaError_t bit_reverse_multigpu(const STORAGE **values, STORAGE **results, unsigned log_count, const cudaStream_t *streams, const int *device_ids,
                                 unsigned log_devices_count);

struct sort_u32_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  unsigned *values;
  unsigned *sorted_values;
  unsigned count;
};

cudaError_t sort_u32(const sort_u32_configuration &configuration);

cudaError_t tear_down();

} // namespace ff
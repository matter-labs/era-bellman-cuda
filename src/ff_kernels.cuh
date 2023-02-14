#pragma once
#include "ff_dispatch_st.cuh"
#include "memory.cuh"
#include <array>

namespace ff {

struct powers_data {
  unsigned fine_log_count;
  unsigned coarse_log_count;
  unsigned fine_mask;
  unsigned coarse_mask;
  fd_q::storage *fine;
  fd_q::storage *coarse;
};

template <typename FD> __device__ __forceinline__ typename FD::storage get_power(const powers_data &data, const unsigned index, bool inverse) {
  typedef typename FD::storage storage;
  const unsigned idx = inverse ? (1u << FD::CONFIG::omega_log_order) - index : index;
  const unsigned fine_idx = idx & data.fine_mask;
  const unsigned coarse_idx = idx >> data.fine_log_count & data.coarse_mask;
  auto coarse = memory::load<storage, memory::ld_modifier::ca>(data.coarse + coarse_idx);
  if (fine_idx == 0)
    return coarse;
  auto fine = memory::load<storage, memory::ld_modifier::ca>(data.fine + fine_idx);
  return FD::mul(fine, coarse);
}

cudaError precompute_powers(fd_q::storage value, fd_q::storage *powers, unsigned offset, unsigned count);

cudaError_t set_up_powers_of_w(unsigned coarse_log_count);

cudaError_t tear_down_powers_of_w();

cudaError_t set_up_powers_of_g_f(unsigned coarse_log_count);

cudaError_t set_up_powers_of_g_i(unsigned coarse_log_count);

cudaError_t tear_down_powers_of_g_f();

cudaError_t tear_down_powers_of_g_i();

cudaError_t get_powers_data_w(powers_data &data);

cudaError_t get_powers_data_g_f(powers_data &data);

cudaError_t get_powers_data_g_i(powers_data &data);

template <class STORAGE> cudaError_t set_value_by_ref(STORAGE *target, const STORAGE *value, unsigned int count, cudaStream_t stream);

template <class STORAGE> cudaError_t set_value_by_val(STORAGE *target, const STORAGE *value, unsigned int count, cudaStream_t stream);

template <class FD>
cudaError_t ax_by_ref(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_by_val(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t a_plus_x_by_ref(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t a_plus_x_by_val(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_plus_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_plus_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                             unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_plus_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                             unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_minus_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_minus_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_minus_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_minus_ay_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_minus_ay_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                              unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t x_mul_y(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_mul_y_by_ref(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                            unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_mul_y_by_val(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                            unsigned count, cudaStream_t stream);

template <class FD> cudaError_t naive_inverse(const typename FD::storage *x, typename FD::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t get_powers_of_w(typename FD::storage *target, unsigned log_degree, unsigned offset, unsigned count, bool inverse, bool bit_reverse,
                            cudaStream_t stream);

template <class FD>
cudaError_t get_powers_of_g(typename FD::storage *target, unsigned log_degree, unsigned offset, unsigned count, bool inverse, bool bit_reverse,
                            cudaStream_t stream);

template <class FD>
cudaError_t omega_shift(const typename FD::storage *values, typename FD::storage *result, unsigned log_degree, unsigned shift, unsigned offset, unsigned count,
                        bool inverse, cudaStream_t stream);

template <class FD>
cudaError_t batch_inverse_per_thread(const typename FD::storage *inputs, typename FD::storage *scratch, typename FD::storage *outputs, unsigned count,
                                     cudaStream_t stream);

template <class STORAGE>
cudaError_t bit_reverse_single(const STORAGE **values, STORAGE **results, unsigned log_count, unsigned partition_id, cudaStream_t stream, unsigned log_split);

template <class STORAGE> cudaError_t select(const STORAGE *source, STORAGE *destination, const unsigned *indexes, unsigned count, cudaStream_t stream);

} // namespace ff

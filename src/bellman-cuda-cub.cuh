#pragma once

#include "ff_dispatch_st.cuh"
namespace common {

cudaError_t sort_keys(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, int num_items, int begin_bit = 0,
                      int end_bit = sizeof(unsigned) * 8, cudaStream_t stream = nullptr, bool debug_synchronous = false);

cudaError_t sort_pairs(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, const unsigned *d_values_in,
                       unsigned *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(unsigned) * 8, cudaStream_t stream = nullptr,
                       bool debug_synchronous = false);

cudaError_t sort_pairs_descending(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out,
                                  const unsigned *d_values_in, unsigned *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(unsigned) * 8,
                                  cudaStream_t stream = nullptr, bool debug_synchronous = false);

cudaError_t run_length_encode(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_in, unsigned *d_unique_out, unsigned *d_counts_out,
                              unsigned *d_num_runs_out, int num_items, cudaStream_t stream = nullptr, bool debug_synchronous = false);

cudaError_t exclusive_sum(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_in, unsigned *d_out, int num_items, cudaStream_t stream = nullptr,
                          bool debug_synchronous = false);

} // namespace common

namespace ff {

cudaError_t sum(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items, cudaStream_t stream = nullptr,
                bool debug_synchronous = false);

cudaError_t inclusive_prefix_product(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items,
                                     cudaStream_t stream = nullptr, bool debug_synchronous = false);

cudaError_t inclusive_prefix_product_reverse(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items,
                                             cudaStream_t stream = nullptr, bool debug_synchronous = false);

} // namespace ff
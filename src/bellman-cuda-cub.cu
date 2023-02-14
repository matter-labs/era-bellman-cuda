#include "bellman-cuda-cub.cuh"
#include "ff_dispatch_st.cuh"
#include <cub/cub.cuh>

namespace common {

using namespace cub;

cudaError_t sort_keys(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, int num_items, int begin_bit,
                      int end_bit, cudaStream_t stream, bool debug_synchronous) {
  return DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream, debug_synchronous);
}

cudaError_t sort_pairs(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out, const unsigned *d_values_in,
                       unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream, bool debug_synchronous) {
  return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream,
                                    debug_synchronous);
}

cudaError_t sort_pairs_descending(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_keys_in, unsigned *d_keys_out,
                                  const unsigned *d_values_in, unsigned *d_values_out, int num_items, int begin_bit, int end_bit, cudaStream_t stream,
                                  bool debug_synchronous) {
  return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit,
                                              end_bit, stream, debug_synchronous);
}

cudaError_t run_length_encode(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_in, unsigned *d_unique_out, unsigned *d_counts_out,
                              unsigned *d_num_runs_out, int num_items, cudaStream_t stream, bool debug_synchronous) {
  return DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream,
                                       debug_synchronous);
}

cudaError_t exclusive_sum(void *d_temp_storage, size_t &temp_storage_bytes, const unsigned *d_in, unsigned *d_out, int num_items, cudaStream_t stream,
                          bool debug_synchronous) {
  return DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
}

} // namespace common

namespace ff {

using namespace cub;

struct fq_add {
  typedef fd_q::storage storage;
  __device__ __forceinline__ storage operator()(const storage &a, const storage &b) const { return fd_q::add(a, b); }
};

struct fq_mul {
  typedef fd_q::storage storage;
  __device__ __forceinline__ storage operator()(const storage &a, const storage &b) const { return fd_q::mul(a, b); }
};

cudaError_t sum(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items, cudaStream_t stream,
                bool debug_synchronous) {
  return DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, fq_add(), fd_q::storage(), stream, debug_synchronous);
}

cudaError_t inclusive_prefix_product(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items,
                                     cudaStream_t stream, bool debug_synchronous) {
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, fq_mul(), num_items, stream, debug_synchronous);
}

cudaError_t inclusive_prefix_product_reverse(void *d_temp_storage, size_t &temp_storage_bytes, const fd_q::storage *d_in, fd_q::storage *d_out, int num_items,
                                             cudaStream_t stream, bool debug_synchronous) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, fq_mul(), num_items, stream, debug_synchronous);
}

} // namespace ff
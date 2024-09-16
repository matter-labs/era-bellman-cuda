#include "allocator.cuh"
#include "bellman-cuda-cub.cuh"
#include "common.cuh"
#include "pn.cuh"
#include <cassert>

namespace pn {

using namespace allocator;
using namespace common;

cudaError_t set_up() {
  HANDLE_CUDA_ERROR(set_up_powers());
  return cudaSuccess;
}

cudaError_t generate_permutation_polynomials(const generate_permutation_polynomials_configuration &cfg) {
  typedef allocation<unsigned> unsigned_ints;
  cudaMemPool_t pool = cfg.mem_pool;
  cudaStream_t stream = cfg.stream;
  unsigned int columns_count = cfg.columns_count;
  unsigned int log_rows_count = cfg.log_rows_count;
  const unsigned cells_count = columns_count << log_rows_count;
  const unsigned bits_count = log2_ceiling(columns_count) + log_rows_count;
  allocation<void> temp_storage;
  size_t temp_storage_bytes = 0;
  unsigned_ints unsorted_keys;
  unsigned_ints unsorted_values;
  unsigned_ints sorted_keys;
  unsigned_ints sorted_values;

  HANDLE_CUDA_ERROR(allocate(unsorted_keys, cells_count, pool, stream));
  switch (columns_count) {
  case 3:
    HANDLE_CUDA_ERROR(transpose<3>(unsorted_keys, cfg.indexes, log_rows_count, stream));
  case 4:
    HANDLE_CUDA_ERROR(transpose<4>(unsorted_keys, cfg.indexes, log_rows_count, stream));
  default:
    assert(columns_count == 3 || columns_count == 4);
  }
  HANDLE_CUDA_ERROR(allocate(unsorted_values, cells_count, pool, stream));
  HANDLE_CUDA_ERROR(fill_transposed_range(unsorted_values, columns_count, log_rows_count, stream));
  HANDLE_CUDA_ERROR(allocate(sorted_keys, cells_count, pool, stream));
  HANDLE_CUDA_ERROR(allocate(sorted_values, cells_count, pool, stream));
  HANDLE_CUDA_ERROR(sort_pairs(temp_storage, temp_storage_bytes, unsorted_keys, sorted_keys, unsorted_values, sorted_values, cells_count, 0, bits_count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(
      sort_pairs(temp_storage, temp_storage_bytes, unsorted_keys, sorted_keys, unsorted_values, sorted_values, cells_count, 0, bits_count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));
  HANDLE_CUDA_ERROR(free(unsorted_keys, stream));
  HANDLE_CUDA_ERROR(free(unsorted_values, stream));

  unsigned_ints unique_indexes;
  HANDLE_CUDA_ERROR(allocate(unique_indexes, cells_count, pool, stream));
  unsigned_ints run_lengths;
  HANDLE_CUDA_ERROR(allocate(run_lengths, cells_count, pool, stream));
  unsigned_ints runs_count;
  HANDLE_CUDA_ERROR(allocate(runs_count, 1, pool, stream));
  HANDLE_CUDA_ERROR(cudaMemsetAsync(run_lengths, 0, sizeof(unsigned) * cells_count, stream));
  temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(run_length_encode(temp_storage, temp_storage_bytes, sorted_keys, unique_indexes, run_lengths, runs_count, cells_count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(run_length_encode(temp_storage, temp_storage_bytes, sorted_keys, unique_indexes, run_lengths, runs_count, cells_count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));
  HANDLE_CUDA_ERROR(free(runs_count, stream));
  HANDLE_CUDA_ERROR(free(unique_indexes, stream));
  HANDLE_CUDA_ERROR(free(sorted_keys, stream));

  unsigned_ints run_offsets;
  HANDLE_CUDA_ERROR(allocate(run_offsets, cells_count, pool, stream));
  temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(exclusive_sum(temp_storage, temp_storage_bytes, run_lengths, run_offsets, cells_count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(exclusive_sum(temp_storage, temp_storage_bytes, run_lengths, run_offsets, cells_count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));

  unsigned_ints run_indexes;
  HANDLE_CUDA_ERROR(allocate(run_indexes, cells_count, pool, stream));
  HANDLE_CUDA_ERROR(cudaMemsetAsync(run_indexes, 0, sizeof(unsigned) * cells_count, stream));
  HANDLE_CUDA_ERROR(mark_ends_of_runs(run_indexes, run_lengths, run_offsets, cells_count, stream));
  temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(exclusive_sum(temp_storage, temp_storage_bytes, run_indexes, run_indexes, cells_count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(exclusive_sum(temp_storage, temp_storage_bytes, run_indexes, run_indexes, cells_count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));

  HANDLE_CUDA_ERROR(
      generate_permutation_matrix(cfg.target, cfg.scalars, sorted_values, run_indexes, run_lengths, run_offsets, columns_count, log_rows_count, stream));

  HANDLE_CUDA_ERROR(free(run_indexes, stream));
  HANDLE_CUDA_ERROR(free(run_offsets, stream));
  HANDLE_CUDA_ERROR(free(run_lengths, stream));
  HANDLE_CUDA_ERROR(free(sorted_values, stream));
  return cudaSuccess;
}

cudaError_t tear_down() { return cudaSuccess; }

} // namespace pn
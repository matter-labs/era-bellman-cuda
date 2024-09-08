#include "common.cuh"
#include "ff_kernels.cuh"
#include "pn_kernels.cuh"
#include <cub/cub.cuh>

namespace pn {

using namespace ff;

__constant__ powers_data powers_data_w;

__device__ __forceinline__ fd_q::storage get_power_of_w(const unsigned index, bool inverse) { return get_power<fd_q>(powers_data_w, index, inverse); }

cudaError_t set_up_powers() {
  powers_data w{};
  HANDLE_CUDA_ERROR(get_powers_data_w(w))
  HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(powers_data_w, &w, sizeof(powers_data)));
  return cudaSuccess;
}

#define BLOCK_SIZE 256
template <unsigned COL_COUNT> __global__ void transpose_kernel(unsigned *dst, const unsigned *src, const unsigned log_rows_count) {
  const unsigned gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (gid >= (1 << log_rows_count))
    return;
  constexpr int warp_threads = 4;
  using warp_store = cub::WarpStore<unsigned, COL_COUNT, cub::WARP_STORE_TRANSPOSE, warp_threads>;
  constexpr unsigned warps_in_block = BLOCK_SIZE / warp_threads;
  constexpr unsigned tile_size = COL_COUNT * warp_threads;
  const unsigned warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  __shared__ typename warp_store::TempStorage temp_storage[warps_in_block];
  unsigned tile_offset = blockIdx.x * warps_in_block + warp_id;
  unsigned thread_data[COL_COUNT];
  for (unsigned i = 0; i < COL_COUNT; i++)
    thread_data[i] = src[gid + (i << log_rows_count)];
  warp_store(temp_storage[warp_id]).Store(dst + tile_offset * tile_size, thread_data);
}

template <unsigned COL_COUNT> cudaError_t transpose(unsigned *dst, const unsigned *src, const unsigned log_rows_count, cudaStream_t stream) {
  const unsigned count = 1 << log_rows_count;
  const dim3 block_dim = count < BLOCK_SIZE ? count : BLOCK_SIZE;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  transpose_kernel<COL_COUNT><<<grid_dim, block_dim, 0, stream>>>(dst, src, log_rows_count);
  return cudaGetLastError();
}

template cudaError_t transpose<4>(unsigned *dst, const unsigned *src, unsigned log_rows_count, cudaStream_t stream);
#undef BLOCK_SIZE

__global__ void fill_transposed_range_kernel(unsigned *values, const unsigned columns_count, const unsigned log_rows_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned count = columns_count << log_rows_count;
  if (gid >= count)
    return;
  const unsigned col = gid % columns_count;
  const unsigned row = gid / columns_count;
  values[gid] = (col << log_rows_count) + row;
}

cudaError_t fill_transposed_range(unsigned *values, const unsigned columns_count, const unsigned log_rows_count, cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const unsigned count = columns_count << log_rows_count;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  fill_transposed_range_kernel<<<grid_dim, block_dim, 0, stream>>>(values, columns_count, log_rows_count);
  return cudaGetLastError();
}

__global__ void mark_ends_of_runs_kernel(unsigned *values, const unsigned *run_lengths, const unsigned *run_offsets, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned run_length = run_lengths[gid];
  if (run_length == 0)
    return;
  const unsigned run_offset = run_offsets[gid];
  values[run_offset + run_length - 1] = 1;
}

cudaError_t mark_ends_of_runs(unsigned *values, const unsigned *run_lengths, const unsigned *run_offsets, unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  mark_ends_of_runs_kernel<<<grid_dim, block_dim, 0, stream>>>(values, run_lengths, run_offsets, count);
  return cudaGetLastError();
}

__global__ void generate_permutation_matrix_kernel(fd_q::storage *values, const fd_q::storage *scalars, const unsigned *cell_indexes,
                                                   const unsigned *run_indexes, const unsigned *run_lengths, const unsigned *run_offsets,
                                                   const unsigned columns_count, const unsigned log_rows_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned count = columns_count << log_rows_count;
  if (gid >= count)
    return;
  const unsigned run_index = run_indexes[gid];
  const unsigned run_length = run_lengths[run_index];
  const unsigned run_offset = run_offsets[run_index];
  const unsigned src_in_run_index = gid - run_offset;
  const unsigned dst_in_run_index = run_index == 0 ? src_in_run_index : (src_in_run_index + run_length - 1) % run_length;
  const unsigned src_cell_index = cell_indexes[run_offset + src_in_run_index];
  const unsigned dst_cell_index = cell_indexes[run_offset + dst_in_run_index];
  const unsigned src_row_index = src_cell_index & ((1 << log_rows_count) - 1);
  const unsigned src_col_index = src_cell_index >> log_rows_count;
  const unsigned shift = fd_q::CONFIG::omega_log_order - log_rows_count;
  const fd_q::storage twiddle = get_power_of_w(src_row_index << shift, false);
  const fd_q::storage scalar = scalars[src_col_index];
  const fd_q::storage value = fd_q::mul(twiddle, scalar);
  memory::store(values + dst_cell_index, value);
}

cudaError_t generate_permutation_matrix(fd_q::storage *values, const fd_q::storage *scalars, const unsigned *cell_indexes, const unsigned *run_indexes,
                                        const unsigned *run_lengths, const unsigned *run_offsets, const unsigned columns_count, const unsigned log_rows_count,
                                        cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const unsigned count = columns_count << log_rows_count;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  generate_permutation_matrix_kernel<<<grid_dim, block_dim, 0, stream>>>(values, scalars, cell_indexes, run_indexes, run_lengths, run_offsets, columns_count,
                                                                         log_rows_count);
  return cudaGetLastError();
}
__global__ void set_values_from_packed_bits_kernel(fd_q::storage *values, const unsigned *packet_bits, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned word_count = (count - 1) / 32 + 1;
  if (gid >= word_count)
    return;
  unsigned word = packet_bits[gid];
  unsigned offset = gid * 32;
  for (unsigned i = 0; i < 32 && offset + i < count; i++) {
    values[offset + i] = (word & 1) ? fd_q::get_one() : fd_q::storage{};
    word >>= 1;
  }
}

cudaError_t set_values_from_packed_bits(fd_q::storage *values, const unsigned *packet_bits, const unsigned count, cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const unsigned word_count = (count - 1) / 32 + 1;
  const dim3 block_dim = word_count < threads_per_block ? word_count : threads_per_block;
  const dim3 grid_dim = (word_count - 1) / block_dim.x + 1;
  set_values_from_packed_bits_kernel<<<grid_dim, block_dim, 0, stream>>>(values, packet_bits, count);
  return cudaGetLastError();
}

__global__ void distribute_values_kernel(const fd_q::storage *src, fd_q::storage *dst, const unsigned count, const unsigned stride) {
  typedef fd_q::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const auto value = memory::load<storage, memory::ld_modifier::cs>(src + gid);
  memory::store<storage, memory::st_modifier::cs>(dst + gid * stride, value);
}

cudaError_t distribute_values(const fd_q::storage *src, fd_q::storage *dst, const unsigned count, const unsigned stride, cudaStream_t stream) {
  const unsigned threads_per_block = 128;
  const dim3 block_dim = count < threads_per_block ? count : threads_per_block;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  distribute_values_kernel<<<grid_dim, block_dim, 0, stream>>>(src, dst, count, stride);
  return cudaGetLastError();
}

} // namespace pn
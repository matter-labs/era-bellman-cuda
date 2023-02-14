#include "common.cuh"
#include "ec.cuh"
#include "msm_kernels.cuh"

namespace msm {

#define MAX_THREADS 128
#define UINT4_COUNT (sizeof(storage) / sizeof(uint4))
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS)
#endif
    __global__ void initialize_buckets_kernel(point_xyzz *buckets, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const auto bucket_index = gid / UINT4_COUNT;
  const auto element_index = gid % UINT4_COUNT;
  auto elements = reinterpret_cast<uint4 *>(&buckets[bucket_index].zz);
  memory::store<uint4, memory::st_modifier::cs>(elements + element_index, {});
}

__host__ cudaError_t initialize_buckets(point_xyzz *buckets, const unsigned count, cudaStream_t stream) {
  auto count_u4 = UINT4_COUNT * count;
  const dim3 block_dim = count_u4 < MAX_THREADS ? count_u4 : MAX_THREADS;
  const dim3 grid_dim = (count_u4 - 1) / block_dim.x + 1;
  initialize_buckets_kernel<<<grid_dim, block_dim, 0, stream>>>(buckets, count_u4);
  return cudaGetLastError();
}
#undef UINT4_COUNT
#undef MAX_THREADS

#define MAX_THREADS 128
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS)
#endif
    __global__ void compute_bucket_indexes_kernel(const fd_q::storage *__restrict__ scalars, const unsigned windows_count, const unsigned window_bits,
                                                  unsigned *__restrict__ bucket_indexes, unsigned *__restrict__ base_indexes, const unsigned count) {
  constexpr unsigned highest_bit_mask = 0x80000000;
  const unsigned scalar_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar_index >= count)
    return;
  const unsigned top_window_unused_bits = windows_count * window_bits - fd_q::MBC;
  const unsigned top_window_unused_mask = (1 << top_window_unused_bits) - 1;
  const unsigned top_window_used_bits = window_bits - top_window_unused_bits;
  const unsigned signed_window_bits = window_bits - 1;
  const unsigned top_window_signed_window_bits = top_window_used_bits - 1;
  const unsigned top_bucket_index = 1 << signed_window_bits;
  const unsigned top_window_top_bucket_index = 1 << top_window_signed_window_bits;
  const storage pos_scalar = fd_q::from_montgomery(memory::load(scalars + scalar_index));
  const storage neg_scalar = fd_q::neg(pos_scalar);
  unsigned global_sign = fd_q::lt(pos_scalar, neg_scalar) ? 0 : highest_bit_mask;
  const storage scalar = global_sign ? neg_scalar : pos_scalar;
  unsigned borrow = 0;
  for (unsigned i = 0; i < windows_count; i++) {
    const unsigned window_index = i;
    const unsigned window_mask = window_index << window_bits;
    unsigned bucket_index = fd_q::extract_bits(scalar, window_index * window_bits, window_bits);
    bucket_index += borrow;
    borrow = 0;
    unsigned sign = global_sign;
    if (bucket_index > top_bucket_index) {
      bucket_index = (top_bucket_index << 1) - bucket_index;
      borrow = 1;
      sign ^= highest_bit_mask;
    }
    bool is_top_window = window_index == windows_count - 1;
    const unsigned zero_mask = bucket_index ? 0 : highest_bit_mask;
    bucket_index = ((bucket_index & ((is_top_window ? top_window_top_bucket_index : top_bucket_index) - 1)) << 1) |
                   (bucket_index >> (is_top_window ? top_window_signed_window_bits : signed_window_bits));
    const unsigned bucket_index_offset = is_top_window ? (scalar_index & top_window_unused_mask) << top_window_used_bits : 0;
    const unsigned output_index = window_index * count + scalar_index;
    bucket_indexes[output_index] = zero_mask | window_mask | bucket_index_offset | bucket_index;
    base_indexes[output_index] = sign | scalar_index;
  }
}

__host__ cudaError_t compute_bucket_indexes(const fd_q::storage *scalars, const unsigned windows_count, const unsigned window_bits, unsigned *bucket_indexes,
                                            unsigned *base_indexes, const unsigned count, cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  compute_bucket_indexes_kernel<<<grid_dim, block_dim, 0, stream>>>(scalars, windows_count, window_bits, bucket_indexes, base_indexes, count);
  return cudaGetLastError();
}
#undef MAX_THREADS

#define MAX_THREADS 128
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS)
#endif
    __global__ void remove_zero_buckets_kernel(const unsigned *unique_bucket_indexes, unsigned *bucket_run_lengths, const unsigned *bucket_runs_count,
                                               const unsigned count) {
  constexpr unsigned highest_bit_mask = 0x80000000;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned runs_count = *bucket_runs_count;
  const unsigned bucket_index = unique_bucket_indexes[gid];
  const bool is_zero = bucket_index & highest_bit_mask;
  if (gid >= runs_count || is_zero)
    bucket_run_lengths[gid] = 0;
}

__host__ cudaError_t remove_zero_buckets(unsigned *unique_bucket_indexes, unsigned *bucket_run_lengths, const unsigned *bucket_runs_count, const unsigned count,
                                         cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  remove_zero_buckets_kernel<<<grid_dim, block_dim, 0, stream>>>(unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, count);
  return cudaGetLastError();
}
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
template <bool IS_FIRST>
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS, MIN_BLOCKS)
#endif
    __global__ void aggregate_buckets_kernel(const unsigned *__restrict__ base_indexes, const unsigned *__restrict__ bucket_run_offsets,
                                             const unsigned *__restrict__ bucket_run_lengths, const unsigned *__restrict__ bucket_indexes,
                                             const point_affine *__restrict__ bases, point_xyzz *__restrict__ buckets, const unsigned count) {
  constexpr unsigned negative_sign = 0x80000000;
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned length = bucket_run_lengths[gid];
  if (length == 0)
    return;
  const unsigned base_indexes_offset = bucket_run_offsets[gid];
  const unsigned *indexes = base_indexes + base_indexes_offset;
  const unsigned bucket_index = bucket_indexes[gid] >> 1;
  point_xyzz bucket;
  if (IS_FIRST) {
    unsigned base_index = *indexes++;
    unsigned sign = base_index & negative_sign;
    base_index &= ~negative_sign;
    auto base = memory::load<point_affine, memory::ld_modifier::g>(bases + base_index);
    if (sign)
      base = point_affine::neg(base, f);
    bucket = point_affine::to_xyzz(base, f);
  } else {
    bucket = memory::load<point_xyzz, memory::ld_modifier::cs>(buckets + bucket_index);
  }
#pragma unroll 1
  for (unsigned i = IS_FIRST ? 1 : 0; i < length; i++) {
    unsigned base_index = *indexes++;
    unsigned sign = base_index & negative_sign;
    base_index &= ~negative_sign;
    auto base = memory::load<point_affine, memory::ld_modifier::g>(bases + base_index);
    if (sign)
      base = point_affine::neg(base, f);
    bucket = curve::add(bucket, base, f);
  }
  memory::store<point_xyzz, memory::st_modifier::cs>(buckets + bucket_index, bucket);
}

__host__ cudaError_t aggregate_buckets(const bool is_first, const unsigned *base_indexes, const unsigned *bucket_run_offsets,
                                       const unsigned *bucket_run_lengths, const unsigned *bucket_indexes, const point_affine *bases, point_xyzz *buckets,
                                       const unsigned count, cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  auto kernel = is_first ? aggregate_buckets_kernel<true> : aggregate_buckets_kernel<false>;
  kernel<<<grid_dim, block_dim, 0, stream>>>(base_indexes, bucket_run_offsets, bucket_run_lengths, bucket_indexes, bases, buckets, count);
  return cudaGetLastError();
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS)
#endif
    __global__ void extract_top_buckets_kernel(point_xyzz *buckets, point_xyzz *top_buckets, const unsigned bits_count, const unsigned windows_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= windows_count)
    return;
  const unsigned bucket_index = gid << bits_count;
  top_buckets[gid] = buckets[bucket_index];
  buckets[bucket_index] = point_xyzz::point_at_infinity(fd_p());
}

__host__ cudaError_t extract_top_buckets(point_xyzz *buckets, point_xyzz *top_buckets, const unsigned bits_count, const unsigned windows_count,
                                         cudaStream_t stream) {
  const dim3 block_dim = windows_count < MAX_THREADS ? windows_count : MAX_THREADS;
  const dim3 grid_dim = (windows_count - 1) / block_dim.x + 1;
  extract_top_buckets_kernel<<<grid_dim, block_dim, 0, stream>>>(buckets, top_buckets, bits_count, windows_count);
  return cudaGetLastError();
}
#undef MAX_THREADS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS, MIN_BLOCKS)
#endif
    __global__ void split_windows_kernel(const unsigned source_window_bits_count, const unsigned source_windows_count,
                                         const point_xyzz *__restrict__ source_buckets, point_xyzz *__restrict__ target_buckets, const unsigned count) {
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned target_window_bits_count = (source_window_bits_count + 1) >> 1;
  const unsigned target_windows_count = source_windows_count << 1;
  const unsigned target_partition_buckets_count = target_windows_count << target_window_bits_count;
  const unsigned target_partitions_count = count / target_partition_buckets_count;
  const unsigned target_partition_index = gid / target_partition_buckets_count;
  const unsigned target_partition_tid = gid % target_partition_buckets_count;
  const unsigned target_window_buckets_count = 1 << target_window_bits_count;
  const unsigned target_window_index = target_partition_tid / target_window_buckets_count;
  const unsigned target_window_tid = target_partition_tid % target_window_buckets_count;
  const unsigned split_index = target_window_index & 1;
  const unsigned source_window_buckets_per_target = source_window_bits_count & 1
                                                        ? split_index ? (target_window_tid >> (target_window_bits_count - 1) ? 0 : target_window_buckets_count)
                                                                      : 1 << (source_window_bits_count - target_window_bits_count)
                                                        : target_window_buckets_count;
  const unsigned source_window_index = target_window_index >> 1;
  const unsigned source_offset = source_window_index << source_window_bits_count;
  const unsigned target_shift = target_window_bits_count * split_index;
  const unsigned target_offset = target_window_tid << target_shift;
  const unsigned global_offset = source_offset + target_offset;
  const unsigned index_mask = (1 << target_shift) - 1;
  point_xyzz target_bucket = point_xyzz::point_at_infinity(f);
#pragma unroll 1
  for (unsigned i = target_partition_index; i < source_window_buckets_per_target; i += target_partitions_count) {
    const unsigned index_offset = i & index_mask | (i & ~index_mask) << target_window_bits_count;
    const unsigned load_offset = global_offset + index_offset;
    const auto source_bucket = memory::load<point_xyzz, memory::ld_modifier::g>(source_buckets + load_offset);
    target_bucket = i == target_partition_index ? source_bucket : curve::add(target_bucket, source_bucket, f);
  }
  memory::store<point_xyzz, memory::st_modifier::cs>(target_buckets + gid, target_bucket);
}

__host__ cudaError_t split_windows(const unsigned source_window_bits_count, const unsigned source_windows_count, const point_xyzz *source_buckets,
                                   point_xyzz *target_buckets, const unsigned count, cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  split_windows_kernel<<<grid_dim, block_dim, 0, stream>>>(source_window_bits_count, source_windows_count, source_buckets, target_buckets, count);
  return cudaGetLastError();
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#define MIN_BLOCKS 16
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS, MIN_BLOCKS)
#endif
    __global__ void reduce_buckets_kernel(point_xyzz *buckets, const unsigned count) {
  const field f = field();
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  buckets += gid;
  const auto a = memory::load<point_xyzz, memory::ld_modifier::g>(buckets);
  const auto b = memory::load<point_xyzz, memory::ld_modifier::g>(buckets + count);
  const point_xyzz result = curve::add(a, b, f);
  memory::store<point_xyzz, memory::st_modifier::cs>(buckets, result);
}

__host__ cudaError_t reduce_buckets(point_xyzz *buckets, const unsigned count, cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  reduce_buckets_kernel<<<grid_dim, block_dim, 0, stream>>>(buckets, count);
  return cudaGetLastError();
}
#undef MAX_THREADS
#undef MIN_BLOCKS

#define MAX_THREADS 32
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS)
#endif
    __global__ void last_pass_gather_kernel(const unsigned bits_count_pass_one, const point_xyzz *__restrict__ source, const point_xyzz *top_buckets,
                                            point_jacobian *__restrict__ target, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const field f = field();
  const unsigned signed_bits_count_pass_one = bits_count_pass_one - 1;
  unsigned window_index = gid / bits_count_pass_one;
  unsigned window_tid = gid % bits_count_pass_one;
  point_xyzz pz;
  if (window_tid == signed_bits_count_pass_one || gid == count - 1) {
    pz = memory::load<point_xyzz, memory::ld_modifier::g>(top_buckets + window_index);
  } else {
    for (unsigned bits_count = signed_bits_count_pass_one; bits_count > 1;) {
      bits_count = (bits_count + 1) >> 1;
      window_index <<= 1;
      if (window_tid >= bits_count) {
        window_index++;
        window_tid -= bits_count;
      }
    }
    const unsigned sid = (window_index << 1) + 1;
    pz = memory::load<point_xyzz, memory::ld_modifier::g>(source + sid);
  }
  const point_jacobian pj = point_xyzz::to_jacobian(pz, f);
  memory::store<point_jacobian, memory::st_modifier::cs>(target + gid, pj);
}

__host__ cudaError_t last_pass_gather(const unsigned bits_count_pass_one, const point_xyzz *source, const point_xyzz *top_buckets, point_jacobian *target,
                                      const unsigned count, cudaStream_t stream) {
  const dim3 block_dim = count < MAX_THREADS ? count : MAX_THREADS;
  const dim3 grid_dim = (count - 1) / block_dim.x + 1;
  last_pass_gather_kernel<<<grid_dim, block_dim, 0, stream>>>(bits_count_pass_one, source, top_buckets, target, count);
  return cudaGetLastError();
}
#undef MAX_THREADS

template <class T> __inline__ __host__ cudaError_t set_kernel_attributes(T *func) {
  HANDLE_CUDA_ERROR(cudaFuncSetCacheConfig(func, cudaFuncCachePreferL1));
  HANDLE_CUDA_ERROR(cudaFuncSetAttribute(func, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
  return cudaSuccess;
}

__host__ cudaError_t set_kernel_attributes() {
  HANDLE_CUDA_ERROR(set_kernel_attributes(initialize_buckets_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(compute_bucket_indexes_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(remove_zero_buckets_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(aggregate_buckets_kernel<false>));
  HANDLE_CUDA_ERROR(set_kernel_attributes(aggregate_buckets_kernel<true>));
  HANDLE_CUDA_ERROR(set_kernel_attributes(extract_top_buckets_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(split_windows_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(reduce_buckets_kernel));
  HANDLE_CUDA_ERROR(set_kernel_attributes(last_pass_gather_kernel));
  return cudaSuccess;
}

} // namespace msm
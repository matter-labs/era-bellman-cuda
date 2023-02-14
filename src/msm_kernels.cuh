#pragma once
#include "ec.cuh"
#include "ff_dispatch_st.cuh"

namespace msm {

typedef ec<fd_p, 3> curve;
typedef curve::storage storage;
typedef curve::field field;
typedef curve::point_affine point_affine;
typedef curve::point_jacobian point_jacobian;
typedef curve::point_xyzz point_xyzz;
typedef curve::point_projective point_projective;

__host__ cudaError_t initialize_buckets(point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t compute_bucket_indexes(const fd_q::storage *scalars, unsigned windows_count, unsigned window_bits, unsigned *bucket_indexes,
                                            unsigned *base_indexes, unsigned count, cudaStream_t stream);

__host__ cudaError_t remove_zero_buckets(unsigned *unique_bucket_indexes, unsigned *bucket_run_lengths, const unsigned *bucket_runs_count, unsigned count,
                                         cudaStream_t stream);

__host__ cudaError_t aggregate_buckets(bool is_first, const unsigned *base_indexes, const unsigned *bucket_run_offsets, const unsigned *bucket_run_lengths,
                                       const unsigned *bucket_indexes, const point_affine *bases, point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t extract_top_buckets(point_xyzz *buckets, point_xyzz *top_buckets, unsigned bits_count, unsigned windows_count, cudaStream_t stream);

__host__ cudaError_t split_windows(unsigned source_window_bits_count, unsigned source_windows_count, const point_xyzz *source_buckets,
                                   point_xyzz *target_buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t reduce_buckets(point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t last_pass_gather(unsigned bits_count_pass_one, const point_xyzz *source, const point_xyzz *top_buckets, point_jacobian *target,
                                      unsigned count, cudaStream_t stream);

__host__ cudaError_t set_kernel_attributes();

} // namespace msm
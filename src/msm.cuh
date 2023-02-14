#pragma once

#include "msm_kernels.cuh"

namespace msm {

cudaError_t set_up();

struct execution_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  point_affine *bases;
  fd_q::storage *scalars;
  point_jacobian *results;
  unsigned log_scalars_count;
  cudaEvent_t h2d_copy_finished;
  cudaHostFn_t h2d_copy_finished_callback;
  void *h2d_copy_finished_callback_data;
  cudaEvent_t d2h_copy_finished;
  cudaHostFn_t d2h_copy_finished_callback;
  void *d2h_copy_finished_callback_data;
  bool force_min_chunk_size;
  unsigned log_min_chunk_size;
  bool force_max_chunk_size;
  unsigned log_max_chunk_size;
};

cudaError_t execute_async(const execution_configuration &configuration);

cudaError_t tear_down();

} // namespace msm

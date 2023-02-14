#pragma once

#include "ff_dispatch_st.cuh"

namespace ntt {

cudaError_t set_up();

// Max number of GPUS across which a single NTT's input and output data may be (evenly) split.
constexpr unsigned LOG_MAX_DATA_SPLIT_GPUS = 3;

struct execution_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  fd_q::storage *inputs;
  fd_q::storage *outputs;
  unsigned log_values_count;
  bool bit_reversed_inputs;
  bool inverse;
  bool can_overwrite_inputs;
  unsigned log_extension_degree;
  unsigned coset_index;
  cudaEvent_t h2d_copy_finished;
  cudaHostFn_t h2d_copy_finished_callback;
  void *h2d_copy_finished_callback_data;
  cudaEvent_t d2h_copy_finished;
  cudaHostFn_t d2h_copy_finished_callback;
  void *d2h_copy_finished_callback_data;
};

cudaError_t execute_async(const execution_configuration &configuration);

cudaError_t execute_async_multigpu(const execution_configuration *configurations, const int *dev_ids, unsigned log_n_devs);

cudaError_t tear_down();

} // namespace ntt

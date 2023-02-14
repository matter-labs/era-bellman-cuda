#include "allocator.cuh"
#include "common.cuh"
#include "ff_dispatch_st.cuh"
#include "ntt.cuh"
#include "ntt_kernels.cuh"
#include "utils.cuh"
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace ntt {

using namespace allocator;

cudaError_t set_up() {
  HANDLE_CUDA_ERROR(set_up_inv_sizes<fd_q>());
  HANDLE_CUDA_ERROR(set_up_powers());
  return cudaSuccess;
}

// Sets sequence of kernels for a particular NTT.
// Specifically, sets the number of stages and forced-adjacent-tile-size for each kernel.
// If a manual specification is not present, uses a simple heuristic and chooses between just 2 tile sizes, 1 and 16.
// This gives solid performance for N=2^28, but the kernel supports other low-power-of-two
// tile sizes, so we can easily tune these heuristics over time/for different devices.
namespace {
constexpr unsigned log_n_max = 28;

struct Opts {
  unsigned stages;
  unsigned log_tile_sz;
};

std::unordered_map<unsigned, std::vector<Opts>> manual_heuristics = [] {
  std::unordered_map<unsigned, std::vector<Opts>> init{};
  init[25] = {{8, 2}, {8, 2}, {9, 0}};              // nonbitrev -> bitrev
  init[25 + log_n_max] = {{9, 0}, {8, 2}, {8, 2}};  // bitrev -> nonbitrev
  init[26] = {{8, 2}, {8, 2}, {10, 0}};             // nonbitrev -> bitrev
  init[26 + log_n_max] = {{10, 0}, {8, 2}, {8, 2}}; // bitrev -> nonbitrev
  return init;
}();
} // namespace

std::vector<launch_config> plan_launches(const unsigned logN, const bool bit_reversed_inputs, fd_q::storage **inputs, fd_q::storage **outputs) {
  assert(logN <= log_n_max);

  std::vector<launch_config> planned_launches;

  unsigned start_stage = 0;

  const auto &maybe_manual_heuristic = manual_heuristics.find(logN + bit_reversed_inputs * log_n_max);
  if (maybe_manual_heuristic != manual_heuristics.end()) {
    const auto &opts = maybe_manual_heuristic->second;
    unsigned kernels = opts.size();
    if (bit_reversed_inputs)
      assert(opts[0].log_tile_sz == 0);
    else
      assert(opts[kernels - 1].log_tile_sz == 0);
    for (unsigned i = 0; i < kernels; i++) {
      auto stages = opts[i].stages;
      auto log_tile_sz = opts[i].log_tile_sz;
      assert(stages + log_tile_sz <= MAX_SMEM_STAGES);
      unsigned elems_per_block = (1 << stages) * (1 << log_tile_sz);
      bool is_first_launch = (i == 0);
      bool is_last_launch = (i == kernels - 1);
      planned_launches.push_back(
          {start_stage, stages, elems_per_block, log_tile_sz, is_first_launch, is_last_launch, is_first_launch ? inputs : outputs, outputs});
      start_stage += stages;
    }
    assert(start_stage == logN);
  } else {
    unsigned stages_remaining = logN;
    unsigned stages_force_1_adjacent = std::min(MAX_SMEM_STAGES, logN);

    if (bit_reversed_inputs) {
      // The first 10 stages act on contiguous data anyway, so we just use a forced-adjacent-tile size of 1.
      unsigned elems_per_block = (1 << stages_force_1_adjacent);
      bool is_last_launch = stages_remaining == stages_force_1_adjacent;
      launch_config cfg = {start_stage, stages_force_1_adjacent, elems_per_block, 0, true, is_last_launch, inputs, outputs};
      planned_launches.push_back(cfg);
      start_stage += stages_force_1_adjacent;
    }

    stages_remaining -= stages_force_1_adjacent;

    while (stages_remaining > 0) {
      unsigned log_tile_sz = 4;
      unsigned stages_this_launch = std::min(MAX_SMEM_STAGES - log_tile_sz, stages_remaining);
      unsigned elems_per_block = (1 << stages_this_launch) * (1 << log_tile_sz);
      bool is_first_launch = !bit_reversed_inputs && stages_remaining == logN - stages_force_1_adjacent;
      bool is_last_launch = bit_reversed_inputs && stages_remaining == stages_this_launch;
      launch_config cfg = {start_stage, stages_this_launch, elems_per_block, log_tile_sz, is_first_launch, is_last_launch, is_first_launch ? inputs : outputs,
                           outputs};
      planned_launches.push_back(cfg);
      start_stage += stages_this_launch;
      stages_remaining -= stages_this_launch;
    }

    if (!bit_reversed_inputs) {
      // The last 10 stages act on contiguous data anyway, so we just use a forced-adjacent-tile size of 1.
      unsigned elems_per_block = (1 << stages_force_1_adjacent);
      bool is_first_launch = stages_remaining == logN - stages_force_1_adjacent;
      launch_config cfg = {start_stage, stages_force_1_adjacent, elems_per_block, 0, is_first_launch, true, is_first_launch ? inputs : outputs, outputs};
      planned_launches.push_back(cfg);
    }
  }

  return planned_launches;
}

// Used by tests, but not by bellman-cuda.cu.
cudaError_t execute_async(const execution_configuration &configuration) {
  int dev;
  HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
  HANDLE_CUDA_ERROR(execute_async_multigpu(&configuration, &dev, 0));
  return cudaSuccess;
  // alternative:
  // return execute_async_multigpu(&configuration, &dev, 0);
}

cudaError_t execute_async_multigpu(const execution_configuration *const configurations, const int *const dev_ids, const unsigned log_n_devs) {
  typedef typename fd_q::storage storage;
  typedef allocation<storage> values;

  const unsigned log_values_count = configurations[0].log_values_count;
  const unsigned values_count = 1 << log_values_count;
  const bool bit_reversed_inputs = configurations[0].bit_reversed_inputs;
  const bool inverse = configurations[0].inverse;
  const unsigned n_devs = 1 << log_n_devs;

  assert(log_n_devs <= LOG_MAX_DATA_SPLIT_GPUS);
  for (int d = 1; d < n_devs; d++) {
    assert(log_values_count == configurations[d].log_values_count);
    assert(bit_reversed_inputs == configurations[d].bit_reversed_inputs);
    assert(inverse == configurations[d].inverse);
  }

  storage *d_inputs[n_devs];
  storage *d_outputs[n_devs];
  values a_values[n_devs];
  cudaStream_t streams[n_devs];
  bool copy_outputs[n_devs];

  device_guard guard;

  for (int d = 0; d < n_devs; d++) {
    HANDLE_CUDA_ERROR(guard.set(dev_ids[d]));
    const auto &configuration = configurations[d];

    cudaMemPool_t pool = configuration.mem_pool;
    cudaStream_t stream = configuration.stream;
    streams[d] = stream;

    cudaPointerAttributes inputs_attributes{};
    cudaPointerAttributes outputs_attributes{};
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&inputs_attributes, configuration.inputs));
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&outputs_attributes, configuration.outputs));
    bool copy_inputs = inputs_attributes.type == cudaMemoryTypeUnregistered || inputs_attributes.type == cudaMemoryTypeHost;
    copy_outputs[d] = outputs_attributes.type == cudaMemoryTypeUnregistered || outputs_attributes.type == cudaMemoryTypeHost;

    cudaEvent_t execution_started_event;
    HANDLE_CUDA_ERROR(cudaEventCreate(&execution_started_event, cudaEventDisableTiming));
    HANDLE_CUDA_ERROR(cudaEventRecord(execution_started_event, stream));

    cudaStream_t copy_stream;
    cudaEvent_t copy_finished_event;

    if (copy_inputs) {
      HANDLE_CUDA_ERROR(cudaStreamCreate(&copy_stream));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(copy_stream, execution_started_event));
      HANDLE_CUDA_ERROR(cudaEventCreate(&copy_finished_event, cudaEventDisableTiming));
      HANDLE_CUDA_ERROR(allocate(a_values[d], values_count, pool, stream));
      d_inputs[d] = a_values[d];
      HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_inputs[d], configuration.inputs, sizeof(storage) * values_count, cudaMemcpyHostToDevice, copy_stream));
      HANDLE_CUDA_ERROR(cudaEventRecord(copy_finished_event, copy_stream));
      if (configuration.h2d_copy_finished)
        HANDLE_CUDA_ERROR(cudaEventRecord(configuration.h2d_copy_finished, copy_stream));
      if (configuration.h2d_copy_finished_callback)
        HANDLE_CUDA_ERROR(cudaLaunchHostFunc(copy_stream, configuration.h2d_copy_finished_callback, configuration.h2d_copy_finished_callback_data));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream, copy_finished_event));
      HANDLE_CUDA_ERROR(cudaEventDestroy(copy_finished_event));
      HANDLE_CUDA_ERROR(cudaStreamDestroy(copy_stream));
    } else
      d_inputs[d] = configuration.inputs;

    if (copy_outputs[d]) {
      if (copy_inputs || configuration.can_overwrite_inputs)
        d_outputs[d] = d_inputs[d];
      else {
        HANDLE_CUDA_ERROR(allocate(a_values[d], values_count, pool, stream));
        d_outputs[d] = a_values[d];
      }
    } else
      d_outputs[d] = configuration.outputs;

    HANDLE_CUDA_ERROR(cudaEventDestroy(execution_started_event));
  }

  // Each per-launch config is the same across devices, to give every kernel symmetric awareness of the device input and output
  // arrays on all participating devices.
  const vector<launch_config> launch_configs = plan_launches(log_values_count, bit_reversed_inputs, d_inputs, d_outputs);

  for (const auto lc : launch_configs) {
    bool needs_cross_dev_exchange = log_n_devs && ((bit_reversed_inputs && lc.is_last_launch) || (!bit_reversed_inputs && lc.is_first_launch));
    if (needs_cross_dev_exchange) {
      HANDLE_CUDA_ERROR(sync_device_streams(dev_ids, streams, log_n_devs));
    }
    for (unsigned d = 0; d < n_devs; d++) {
      HANDLE_CUDA_ERROR(guard.set(dev_ids[d]));
      HANDLE_CUDA_ERROR(launch_ntt_shared_memory_stages(lc, log_values_count, bit_reversed_inputs, inverse, configurations[d].log_extension_degree,
                                                        configurations[d].coset_index, log_n_devs, d /*Note: NOT dev_ids[d]*/, streams[d]));
    }
    if (needs_cross_dev_exchange) {
      HANDLE_CUDA_ERROR(sync_device_streams(dev_ids, streams, log_n_devs));
    }
  }

  for (unsigned d = 0; d < n_devs; d++) {
    HANDLE_CUDA_ERROR(guard.set(dev_ids[d]));
    const auto &configuration = configurations[d];
    if (copy_outputs[d]) {
      HANDLE_CUDA_ERROR(cudaMemcpyAsync(configuration.outputs, d_outputs[d], sizeof(storage) * values_count, cudaMemcpyDeviceToHost, streams[d]));
      if (a_values[d].ptr != nullptr)
        HANDLE_CUDA_ERROR(free(a_values[d], streams[d]));
      if (configuration.d2h_copy_finished)
        HANDLE_CUDA_ERROR(cudaEventRecord(configuration.d2h_copy_finished, streams[d]));
      if (configuration.d2h_copy_finished_callback)
        HANDLE_CUDA_ERROR(cudaLaunchHostFunc(streams[d], configuration.d2h_copy_finished_callback, configuration.d2h_copy_finished_callback_data));
    }
  }

  HANDLE_CUDA_ERROR(guard.reset());

  return cudaSuccess;
}

cudaError_t tear_down() {
  HANDLE_CUDA_ERROR(tear_down_inv_sizes<fd_q>());
  return cudaSuccess;
}

} // namespace ntt

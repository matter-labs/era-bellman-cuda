#pragma once

#include "ff_dispatch_st.cuh"

namespace ntt {

template <typename FD> cudaError_t set_up_inv_sizes();

template <typename FD> cudaError_t tear_down_inv_sizes();

cudaError_t set_up_powers();

constexpr uint32_t MAX_SMEM_STAGES = 10;

struct launch_config {
  uint32_t start_stage{};
  uint32_t stages_this_launch{};
  uint32_t elems_per_block{};
  uint32_t log_adjacent_elems_tile_sz{};
  bool is_first_launch = false;
  bool is_last_launch = false;
  fd_q::storage **inputs{};
  fd_q::storage **outputs{};
};

// Note: "dev" is not a cuda device id, it's the virtual index in the group of devs assigned to run the present (potentially multigpu) NTT.
// launch_ntt_shared_memory_stages assumes the caller has already cudaSetDevice-d onto the corresponding device.
cudaError_t launch_ntt_shared_memory_stages(const launch_config &lc, unsigned log_n, bool bit_reversed_inputs, bool inverse, unsigned log_extension_degree,
                                            unsigned coset_index, unsigned log_n_devs, unsigned dev_in_group, cudaStream_t stream);

} // namespace ntt

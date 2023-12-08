#include "common.cuh"
#include "ff_kernels.cuh"
#include "memory.cuh"
#include "ntt.cuh"
#include "ntt_kernels.cuh"
#include <cassert>
#include <vector>

namespace ntt {

using namespace ff;

// We can't just make a member of ff_config_q (for example) because
// "The __device__, __shared__, __managed__ and __constant__ memory space specifiers are not allowed on:
//  - class, struct, and union data members..."
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-specifiers
__constant__ fd_q::storage ff_config_q_inv_sizes[29];

// We only have inv_sizes for fd_q, so we should arrange compile time errors if people try to
// instantiate consumers with an FD other than fd_q.
// https://stackoverflow.com/questions/33413960/force-a-compile-time-error-in-a-template-specialization
template <typename T> struct assert_false : std::false_type {};

template <typename FD> cudaError_t set_up_inv_sizes() {
  static_assert(assert_false<FD>::value);
  return cudaSuccess;
}

template <> cudaError_t set_up_inv_sizes<fd_q>() {
  typedef typename fd_q::storage storage;

  storage inv_sizes_host[29];

  inv_sizes_host[0] = fd_q::CONFIG::one;
  inv_sizes_host[1] = fd_q::CONFIG::two_inv;
  for (int i = 2; i < 29; i++)
    inv_sizes_host[i] = fd_q::mul(inv_sizes_host[i - 1], inv_sizes_host[1]);

  return cudaMemcpyToSymbol(ff_config_q_inv_sizes, inv_sizes_host, sizeof(fd_q::storage) * 29);
}

template <typename FD> __device__ typename FD::storage get_inv_size(uint32_t n) { static_assert(assert_false<FD>::value); }

template <> __device__ typename fd_q::storage get_inv_size<fd_q>(uint32_t n) { return ff_config_q_inv_sizes[n]; }

template <typename FD> cudaError_t tear_down_inv_sizes() {
  static_assert(assert_false<FD>::value);
  return cudaSuccess;
}

template <> cudaError_t tear_down_inv_sizes<fd_q>() { return cudaSuccess; }

__constant__ powers_data powers_data_w;
__constant__ powers_data powers_data_g_f;
__constant__ powers_data powers_data_g_i;

template <typename FD> __device__ __forceinline__ typename FD::storage get_power_of_w(const unsigned index, bool inverse) {
  return get_power<FD>(powers_data_w, index, inverse);
}

template <typename FD> __device__ __forceinline__ typename FD::storage get_power_of_g(const unsigned index, bool inverse) {
  return inverse ? get_power<FD>(powers_data_g_i, index, false) : get_power<FD>(powers_data_g_f, index, false);
}

cudaError_t set_up_powers() {
  powers_data w{};
  powers_data g_f{};
  powers_data g_i{};
  HANDLE_CUDA_ERROR(get_powers_data_w(w))
  HANDLE_CUDA_ERROR(get_powers_data_g_f(g_f));
  HANDLE_CUDA_ERROR(get_powers_data_g_i(g_i));
  HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(powers_data_w, &w, sizeof(powers_data)));
  HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(powers_data_g_f, &g_f, sizeof(powers_data)));
  HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(powers_data_g_i, &g_i, sizeof(powers_data)));
  return cudaSuccess;
}

/*
 * Cooley-Tukey kernels.
 *
 * Q. Why Cooley-Tukey, and not some higher-radix algorithm?
 * A. In-placeness is essential.
 *    Also, we want the ability to go from bitreversed inputs to
 *    non-bitreversed outputs and vice versa.
 *    C-T can straightforwardly meet both requirements.
 *
 *    Cooley-Tukey does not sacrifice performance.
 *    Using careful shared memory marshaling, the kernels below carry out
 *    up to 10 exchange stages at a time for each gmem read+write of the data.
 */

template <typename FD> struct per_device_storage {
  typename FD::storage *data[1 << LOG_MAX_DATA_SPLIT_GPUS];
};

// n_devs must be a power of 2
template <typename FD, bool is_multigpu>
__device__ __forceinline__ typename FD::storage *index_to_addr(const per_device_storage<FD> &addrs, const unsigned idx, const int log_n, const int log_n_devs) {
  if (is_multigpu) {
    const unsigned dev_residence = idx >> (log_n - log_n_devs);
    const unsigned offset = idx & ((1 << (log_n - log_n_devs)) - 1);
#pragma unroll
    for (unsigned i = 0; i < (1 << LOG_MAX_DATA_SPLIT_GPUS); i++)
      if (dev_residence == i)
        return addrs.data[i] + offset;
    // Q: Why the switch statement, as opposed to just
    // return addrs.data[dev_residence] + offset; ?
    // A: "return addrs.data[dev_residence] + offset;" incurred local memory use, causing ~10% slowdown. I don't know why:
    // "addrs" passed from ntt_smem_stages_kernel should be in constant memory, which is dynamically indexable.
    // I guess nvcc moved ntt_smem_stages_kernel "inputs" and "outputs" to registers then tried to dynamically
    // index addr.data here in index_to_addr. Smart :eyeroll: Whatever, switch statement works.
  } else {
    return addrs.data[0] + idx;
  }
}

// Carries out up to MAX_SMEM_STAGES - log_tile_sz C-T stages in shared memory.
// Forces sets of adjacent threads to act on tiles of adjacent data elems,
// which improves cache friendliness.
#define MAX_THREADS 1024
#define MIN_BLOCKS 1
template <typename FD, bool is_multigpu, uint32_t log_tile_sz, bool bit_reversed_inputs>
#ifndef __CUDACC_DEBUG__
__launch_bounds__(MAX_THREADS, MIN_BLOCKS)
#endif
    __global__ void ntt_smem_stages_kernel(const per_device_storage<FD> inputs, const per_device_storage<FD> outputs, const uint32_t start_stage,
                                           const uint32_t stages_this_launch, const unsigned log_n, const bool is_first_launch, const bool is_last_launch,
                                           const bool inverse, const unsigned log_extension_degree, const unsigned coset_index, const unsigned log_n_devs,
                                           const unsigned dev) {
  typedef typename FD::storage storage;

  const uint32_t global_block_idx = blockIdx.x + gridDim.x * dev;

  constexpr uint32_t tile_sz = 1 << log_tile_sz;

  extern __shared__ storage smem[];

  uint32_t gmem_load_stride;
  uint32_t gmem_region_size;
  uint32_t k;
  uint32_t l_tile;
  uint32_t r_tile;
  if (bit_reversed_inputs) {
    gmem_load_stride = 1 << start_stage;
    gmem_region_size = gmem_load_stride * (1 << stages_this_launch);
    l_tile = 2 * (threadIdx.x >> log_tile_sz);
    r_tile = l_tile + 1;
  } else {
    gmem_region_size = 1 << (log_n - start_stage);
    gmem_load_stride = gmem_region_size >> stages_this_launch;
    l_tile = threadIdx.x >> log_tile_sz;
    r_tile = l_tile + (1 << (stages_this_launch - 1));
  }
  const uint32_t blocks_per_gmem_region = gmem_load_stride >> log_tile_sz;
  const uint32_t gmem_region = global_block_idx / blocks_per_gmem_region;
  const uint32_t block_start_in_gmem_region = tile_sz * (global_block_idx & (blocks_per_gmem_region - 1));
  const uint32_t block_start = gmem_region * gmem_region_size + block_start_in_gmem_region;
  uint32_t rank_in_tile = (threadIdx.x & (tile_sz - 1));

  // First stage for this launch. If tile_sz is 1, loads be strided, but 32B each.
  const unsigned l_input_index = block_start + l_tile * gmem_load_stride + rank_in_tile;
  auto E = memory::load<storage, memory::ld_modifier::cs>(index_to_addr<FD, is_multigpu>(inputs, l_input_index, log_n, log_n_devs));
  if (is_first_launch && !inverse && log_extension_degree) {
    const unsigned l_idx = bit_reversed_inputs ? __brev(l_input_index) >> (32 - log_n) : l_input_index;
    if (coset_index) {
      const unsigned shift = FD::CONFIG::omega_log_order - log_n - log_extension_degree;
      const unsigned offset = coset_index << shift;
      auto l_power_of_w = get_power_of_w<FD>(l_idx * offset, false);
      E = FD::template mul<0>(E, l_power_of_w);
    }
    auto l_power_of_g = get_power_of_g<FD>(l_idx, false);
    E = FD::template mul<0>(E, l_power_of_g);
  }

  const unsigned r_input_index = block_start + r_tile * gmem_load_stride + rank_in_tile;
  auto O = memory::load<storage, memory::ld_modifier::cs>(index_to_addr<FD, is_multigpu>(inputs, r_input_index, log_n, log_n_devs));
  if (is_first_launch && !inverse && log_extension_degree) {
    const unsigned r_idx = bit_reversed_inputs ? __brev(r_input_index) >> (32 - log_n) : r_input_index;
    if (coset_index) {
      const unsigned shift = FD::CONFIG::omega_log_order - log_n - log_extension_degree;
      const unsigned offset = coset_index << shift;
      auto r_power_of_w = get_power_of_w<FD>(r_idx * offset, false);
      O = FD::template mul<0>(O, r_power_of_w);
    }
    auto r_power_of_g = get_power_of_g<FD>(r_idx, false);
    O = FD::template mul<0>(O, r_power_of_g);
  }

  if (start_stage > 0) {
    const unsigned shift = FD::CONFIG::omega_log_order - (start_stage + 1);
    k = bit_reversed_inputs ? block_start_in_gmem_region + rank_in_tile : __brev(gmem_region) >> (32 - start_stage);
    const auto twiddle = get_power_of_w<FD>(k << shift, inverse);
    O = FD::template mul<0>(O, twiddle);
  }
  storage E_tmp = E;
  E = FD::template add<2>(E_tmp, O);
  O = FD::template sub<2>(E_tmp, O);

  // Remaining stages for this launch
  if (stages_this_launch > 1) {
    uint32_t l = l_tile * tile_sz + rank_in_tile;
    uint32_t r = r_tile * tile_sz + rank_in_tile;
    for (uint32_t smem_stage = 1; smem_stage < stages_this_launch; smem_stage++) {
      memory::store(smem + l, E);
      memory::store(smem + r, O);

      uint32_t smem_bfly_stride;
      uint32_t smem_bfly_region_size;
      uint32_t smem_region;
      if (bit_reversed_inputs) {
        smem_bfly_stride = 1 << (smem_stage + log_tile_sz);
        smem_bfly_region_size = smem_bfly_stride * 2;
        smem_region = threadIdx.x >> (smem_stage + log_tile_sz); // threadIdx.x / smem_bfly_stride;
        if (smem_bfly_stride > 32)
          __syncthreads();
        else
          __syncwarp();
      } else {
        const uint32_t tmp = stages_this_launch - smem_stage + log_tile_sz;
        smem_bfly_region_size = 1 << tmp;
        smem_bfly_stride = smem_bfly_region_size >> 1;
        smem_region = threadIdx.x >> (tmp - 1); // threadIdx.x / smem_bfly_stride;
        if (smem_bfly_stride >= 32)
          __syncthreads();
        else
          __syncwarp();
      }
      const uint32_t id_in_region = threadIdx.x & (smem_bfly_stride - 1);
      const unsigned shift = FD::CONFIG::omega_log_order - (start_stage + smem_stage + 1);
      if (bit_reversed_inputs) {
        k = (id_in_region >> log_tile_sz) * gmem_load_stride + block_start_in_gmem_region + (id_in_region & (tile_sz - 1)); // oof
      } else {
        const uint32_t smem_bfly_regions = 1 << smem_stage;
        k = __brev(smem_region + gmem_region * smem_bfly_regions) >> (32 - start_stage - smem_stage);
      }
      l = smem_region * smem_bfly_region_size + id_in_region;
      r = l + smem_bfly_stride;

      O = memory::load(smem + r);
      const auto twiddle = get_power_of_w<FD>(k << shift, inverse);
      O = FD::template mul<0>(O, twiddle);
      E_tmp = memory::load(smem + l);
      E = FD::template add<2>(E_tmp, O);
      O = FD::template sub<2>(E_tmp, O);
    }

    // No need for another syncthreads if every thread stores the last two elements it acted on.
    // Depending on layout and start_stage, stores may be widely strided, but 32B each.
    l_tile = l >> log_tile_sz;
    r_tile = r >> log_tile_sz;
    rank_in_tile = l & (tile_sz - 1);
  }

  const auto inv_size = is_last_launch && inverse ? get_inv_size<FD>(log_n) : FD::storage();

  const unsigned l_output_index = block_start + l_tile * gmem_load_stride + rank_in_tile;
  if (is_last_launch) {
    if (inverse) {
      E = FD::template mul<0>(E, inv_size);
      if (log_extension_degree) {
        const unsigned l_idx = bit_reversed_inputs ? l_output_index : __brev(l_output_index) >> (32 - log_n);
        if (coset_index) {
          const unsigned shift = FD::CONFIG::omega_log_order - log_n - log_extension_degree;
          const unsigned offset = coset_index << shift;
          auto l_power_of_w = get_power_of_w<FD>(l_idx * offset, true);
          E = FD::template mul<0>(E, l_power_of_w);
        }
        auto l_power_of_g = get_power_of_g<FD>(l_idx, true);
        E = FD::template mul<0>(E, l_power_of_g);
      }
    }
    E = FD::reduce(E);
  }
  memory::store<storage, memory::st_modifier::cs>(index_to_addr<FD, is_multigpu>(outputs, l_output_index, log_n, log_n_devs), E);

  const unsigned r_output_index = block_start + r_tile * gmem_load_stride + rank_in_tile;
  if (is_last_launch) {
    if (inverse) {
      O = FD::template mul<0>(O, inv_size);
      if (log_extension_degree) {
        const unsigned r_idx = bit_reversed_inputs ? r_output_index : __brev(r_output_index) >> (32 - log_n);
        if (coset_index) {
          const unsigned shift = FD::CONFIG::omega_log_order - log_n - log_extension_degree;
          const unsigned offset = coset_index << shift;
          auto r_power_of_w = get_power_of_w<FD>(r_idx * offset, true);
          O = FD::template mul<0>(O, r_power_of_w);
        }
        auto r_power_of_g = get_power_of_g<FD>(r_idx, true);
        O = FD::template mul<0>(O, r_power_of_g);
      }
    }
    O = FD::reduce(O);
  }
  memory::store<storage, memory::st_modifier::cs>(index_to_addr<FD, is_multigpu>(outputs, r_output_index, log_n, log_n_devs), O);
}
#undef MAX_THREADS
#undef MIN_BLOCKS

template <typename FD, bool is_multigpu, uint32_t log_tile_sz>
void ntt_smem_stages_kernel(const per_device_storage<FD> inputs, const per_device_storage<FD> outputs, const uint32_t start_stage,
                            const uint32_t stages_this_launch, const unsigned log_n, const bool is_first_launch, const bool is_last_launch, const bool inverse,
                            const unsigned log_extension_degree, const unsigned coset_index, const dim3 grid_dim, const dim3 block_dim,
                            const size_t shared_size, cudaStream_t stream, const bool bit_reversed_inputs, const unsigned log_n_devs, const unsigned dev) {
  if (bit_reversed_inputs)
    ntt_smem_stages_kernel<fd_q, is_multigpu, log_tile_sz, true><<<grid_dim, block_dim, shared_size, stream>>>(
        inputs, outputs, start_stage, stages_this_launch, log_n, is_first_launch, is_last_launch, inverse, log_extension_degree, coset_index, log_n_devs, dev);
  else
    ntt_smem_stages_kernel<fd_q, is_multigpu, log_tile_sz, false><<<grid_dim, block_dim, shared_size, stream>>>(
        inputs, outputs, start_stage, stages_this_launch, log_n, is_first_launch, is_last_launch, inverse, log_extension_degree, coset_index, log_n_devs, dev);
}

template <typename FD, bool is_multigpu>
cudaError_t ntt_smem_stages_kernel(const per_device_storage<FD> inputs, const per_device_storage<FD> outputs, const uint32_t start_stage,
                                   const uint32_t stages_this_launch, const unsigned log_n, const bool is_first_launch, const bool is_last_launch,
                                   const bool inverse, const unsigned log_extension_degree, const unsigned coset_index, const dim3 grid_dim,
                                   const dim3 block_dim, const size_t shared_size, cudaStream_t stream, const bool bit_reversed_inputs,
                                   const unsigned log_n_devs, const unsigned dev, const uint32_t log_tile_sz) {
  switch (log_tile_sz) {
  case 0:
    ntt_smem_stages_kernel<fd_q, is_multigpu, 0>(inputs, outputs, start_stage, stages_this_launch, log_n, is_first_launch, is_last_launch, inverse,
                                                 log_extension_degree, coset_index, grid_dim, block_dim, shared_size, stream, bit_reversed_inputs, log_n_devs,
                                                 dev);
    break;
  case 2:
    ntt_smem_stages_kernel<fd_q, is_multigpu, 2>(inputs, outputs, start_stage, stages_this_launch, log_n, is_first_launch, is_last_launch, inverse,
                                                 log_extension_degree, coset_index, grid_dim, block_dim, shared_size, stream, bit_reversed_inputs, log_n_devs,
                                                 dev);
    break;
  case 4:
    ntt_smem_stages_kernel<fd_q, is_multigpu, 4>(inputs, outputs, start_stage, stages_this_launch, log_n, is_first_launch, is_last_launch, inverse,
                                                 log_extension_degree, coset_index, grid_dim, block_dim, shared_size, stream, bit_reversed_inputs, log_n_devs,
                                                 dev);
    break;
  default:
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// Note: "dev" is not a cuda device id, it's the virtual index in the group of devs assigned to run the present (potentially multigpu) NTT.
// launch_ntt_shared_memory_stages assumes the caller has already cudaSetDevice-d onto the corresponding device.
cudaError_t launch_ntt_shared_memory_stages(const launch_config &lc, const unsigned log_n, const bool bit_reversed_inputs, const bool inverse,
                                            const unsigned log_extension_degree, const unsigned coset_index, const unsigned log_n_devs, const unsigned dev,
                                            cudaStream_t stream) {
  const dim3 grid_dim = (1 << (log_n - log_n_devs)) / lc.elems_per_block;
  // The following assert should fire if we try to run a small multigpu NTT (total size < num devs * max smem elems per block (1024))
  // which is a technically legitimate but not very practical scenario.
  assert(grid_dim.x > 0);
  const dim3 block_dim = lc.elems_per_block / 2;
  size_t shared_size = lc.elems_per_block * sizeof(fd_q::storage);
  // this is a little clumsy, maybe better to make PerDeviceStorages members of launch_config?
  per_device_storage<fd_q> inputs{};
  per_device_storage<fd_q> outputs{};
  for (int i = 0; i < (1 << log_n_devs); i++) {
    inputs.data[i] = lc.inputs[i];
    outputs.data[i] = lc.outputs[i];
  }
  if (log_n_devs)
    return ntt_smem_stages_kernel<fd_q, true>(inputs, outputs, lc.start_stage, lc.stages_this_launch, log_n, lc.is_first_launch, lc.is_last_launch, inverse,
                                              log_extension_degree, coset_index, grid_dim, block_dim, shared_size, stream, bit_reversed_inputs, log_n_devs, dev,
                                              lc.log_adjacent_elems_tile_sz);
  else
    return ntt_smem_stages_kernel<fd_q, false>(inputs, outputs, lc.start_stage, lc.stages_this_launch, log_n, lc.is_first_launch, lc.is_last_launch, inverse,
                                               log_extension_degree, coset_index, grid_dim, block_dim, shared_size, stream, bit_reversed_inputs, log_n_devs,
                                               dev, lc.log_adjacent_elems_tile_sz);
}

} // namespace ntt

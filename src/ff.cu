#include "allocator.cuh"
#include "bellman-cuda-cub.cuh"
#include "common.cuh"
#include "ff.cuh"
#include "ff_dispatch_st.cuh"
#include "utils.cuh"

namespace ff {

using namespace allocator;

cudaError_t set_up(const unsigned powers_of_w_coarse_log_count, const unsigned powers_of_g_coarse_log_count) {
  HANDLE_CUDA_ERROR(set_up_powers_of_w(powers_of_w_coarse_log_count));
  HANDLE_CUDA_ERROR(set_up_powers_of_g_f(powers_of_g_coarse_log_count));
  HANDLE_CUDA_ERROR(set_up_powers_of_g_i(powers_of_g_coarse_log_count));
  return cudaSuccess;
}

template <class STORAGE> cudaError_t set_value(STORAGE *target, const STORAGE *value, const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes value_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&value_attributes, value));
  return value_attributes.type == cudaMemoryTypeDevice ? set_value_by_ref(target, value, count, stream) : set_value_by_val(target, value, count, stream);
}

template cudaError_t set_value<ff_storage<8>>(ff_storage<8> *target, const ff_storage<8> *value, const unsigned count, cudaStream_t stream);

template <class STORAGE> cudaError_t set_value_zero(STORAGE *target, const unsigned count, cudaStream_t stream) {
  return cudaMemsetAsync(target, 0, sizeof(STORAGE) * count, stream);
}

template cudaError_t set_value_zero<ff_storage<8>>(ff_storage<8> *target, const unsigned count, cudaStream_t stream);

template <class FD> cudaError_t set_value_one(typename FD::storage *target, const unsigned count, cudaStream_t stream) {
  return set_value_by_val(target, &FD::CONFIG::one, count, stream);
}

template cudaError_t set_value_one<fd_q>(fd_q::storage *target, const unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? ax_by_ref<FD>(a, x, result, count, stream) : ax_by_val<FD>(a, x, result, count, stream);
}

template cudaError_t ax<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t a_plus_x(const typename FD::storage *a, const typename FD::storage *x, typename FD::storage *result, const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? a_plus_x_by_ref<FD>(a, x, result, count, stream) : a_plus_x_by_val<FD>(a, x, result, count, stream);
}

template cudaError_t a_plus_x<fd_q>(const fd_q::storage *a, const fd_q::storage *x, fd_q::storage *result, unsigned count, cudaStream_t stream);

template <class FD>
cudaError_t ax_plus_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                      const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? ax_plus_y_by_ref<FD>(a, x, y, result, count, stream)
                                                   : ax_plus_y_by_val<FD>(a, x, y, result, count, stream);
}

template cudaError_t ax_plus_y<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, unsigned count,
                                     cudaStream_t stream);

template <class FD>
cudaError_t ax_minus_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                       const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? ax_minus_y_by_ref<FD>(a, x, y, result, count, stream)
                                                   : ax_minus_y_by_val<FD>(a, x, y, result, count, stream);
}

template cudaError_t ax_minus_y<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, unsigned count,
                                      cudaStream_t stream);

template <class FD>
cudaError_t x_minus_ay(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                       const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? x_minus_ay_by_ref<FD>(a, x, y, result, count, stream)
                                                   : x_minus_ay_by_val<FD>(a, x, y, result, count, stream);
}

template cudaError_t x_minus_ay<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, unsigned count,
                                      cudaStream_t stream);

template <class FD>
cudaError_t ax_mul_y(const typename FD::storage *a, const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result,
                     const unsigned count, cudaStream_t stream) {
  cudaPointerAttributes a_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&a_attributes, a));
  return a_attributes.type == cudaMemoryTypeDevice ? ax_mul_y_by_ref<FD>(a, x, y, result, count, stream) : ax_mul_y_by_val<FD>(a, x, y, result, count, stream);
}

template cudaError_t ax_mul_y<fd_q>(const fd_q::storage *a, const fd_q::storage *x, const fd_q::storage *y, fd_q::storage *result, unsigned count,
                                    cudaStream_t stream);

cudaError_t grand_product(const grand_product_configuration &configuration) {
  typedef ff_storage<8> storage;
  cudaMemPool_t pool = configuration.mem_pool;
  cudaStream_t stream = configuration.stream;
  const storage *inputs = configuration.inputs;
  storage *outputs = configuration.outputs;
  const unsigned count = configuration.count;
  allocation<void> temp_storage;
  size_t temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(inclusive_prefix_product(temp_storage, temp_storage_bytes, inputs, outputs, count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(inclusive_prefix_product(temp_storage, temp_storage_bytes, inputs, outputs, count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));
  return cudaSuccess;
}

cudaError_t grand_product_reverse(const grand_product_configuration &configuration) {
  typedef ff_storage<8> storage;
  cudaMemPool_t pool = configuration.mem_pool;
  cudaStream_t stream = configuration.stream;
  const storage *inputs = configuration.inputs;
  storage *outputs = configuration.outputs;
  const unsigned count = configuration.count;
  allocation<void> temp_storage;
  size_t temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(inclusive_prefix_product_reverse(temp_storage, temp_storage_bytes, inputs, outputs, count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(inclusive_prefix_product_reverse(temp_storage, temp_storage_bytes, inputs, outputs, count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));
  return cudaSuccess;
}

cudaError_t multiply_by_powers(const multiply_by_powers_configuration &configuration) {
  typedef ff_storage<8> storage;
  cudaMemPool_t pool = configuration.mem_pool;
  cudaStream_t stream = configuration.stream;
  const unsigned count = configuration.count;
  ff_storage<8> *values = configuration.values;
  ff_storage<8> *result = configuration.result;
  const bool use_result = values != result;
  ff_storage<8> *gp_temp;
  allocation<storage> temp;
  if (use_result)
    gp_temp = result;
  else {
    HANDLE_CUDA_ERROR(allocate(temp, count, pool, stream));
    gp_temp = temp;
  }
  HANDLE_CUDA_ERROR(set_value_one<fd_q>(gp_temp, 1, stream));
  if (count > 1) {
    HANDLE_CUDA_ERROR(set_value(gp_temp + 1, configuration.base, count - 1, stream));
    const grand_product_configuration gpc = {pool, stream, gp_temp + 1, gp_temp + 1, count - 1};
    HANDLE_CUDA_ERROR(grand_product(gpc));
  }
  HANDLE_CUDA_ERROR(x_mul_y<fd_q>(values, gp_temp, result, count, stream));
  if (!use_result) {
    HANDLE_CUDA_ERROR(free(temp, stream));
  }
  return cudaSuccess;
}

cudaError_t inverse(const inverse_configuration &configuration) {
  typedef ff_storage<8> storage;
  cudaMemPool_t pool = configuration.mem_pool;
  cudaStream_t stream = configuration.stream;
  storage *inputs = configuration.inputs;
  storage *outputs = configuration.outputs;
  const unsigned count = configuration.count;
  if (outputs == inputs) {
    allocation<storage> scratch;
    HANDLE_CUDA_ERROR(allocate(scratch, count, pool, stream));
    HANDLE_CUDA_ERROR(batch_inverse_per_thread<fd_q>(inputs, scratch, inputs, count, stream));
    HANDLE_CUDA_ERROR(free(scratch, stream));
  } else {
    HANDLE_CUDA_ERROR(batch_inverse_per_thread<fd_q>(inputs, outputs, outputs, count, stream));
  }
  return cudaSuccess;
}

cudaError_t poly_evaluate(const poly_evaluate_configuration &configuration) {
  typedef ff_storage<8> storage;
  cudaMemPool_t pool = configuration.mem_pool;
  cudaStream_t stream = configuration.stream;
  const unsigned count = configuration.count;
  ff_storage<8> *result = configuration.result;
  allocation<storage> temp;
  HANDLE_CUDA_ERROR(allocate(temp, count, pool, stream));
  const multiply_by_powers_configuration mpc = {pool, stream, configuration.values, configuration.point, temp, count};
  HANDLE_CUDA_ERROR(multiply_by_powers(mpc));
  allocation<void> temp_storage;
  size_t temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(sum(temp_storage, temp_storage_bytes, temp.ptr, result, count));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, pool, stream));
  HANDLE_CUDA_ERROR(sum(temp_storage, temp_storage_bytes, temp.ptr, result, count, stream));
  HANDLE_CUDA_ERROR(free(temp_storage, stream));
  HANDLE_CUDA_ERROR(free(temp, stream));
  return cudaSuccess;
}

template <class STORAGE> cudaError_t bit_reverse(const STORAGE *values, STORAGE *result, const unsigned log_count, cudaStream_t stream) {
  return bit_reverse_single(&values, &result, log_count, 0, stream, 0);
}

template cudaError_t bit_reverse(const fd_q::storage *values, fd_q::storage *result, const unsigned log_count, cudaStream_t stream);

template <class STORAGE>
cudaError_t bit_reverse_multigpu(const STORAGE **values, STORAGE **results, const unsigned log_count, const cudaStream_t *streams, const int *device_ids,
                                 const unsigned log_devices_count) {
  HANDLE_CUDA_ERROR(sync_device_streams(device_ids, streams, log_devices_count));
  const unsigned devices_count = 1 << log_devices_count;
  device_guard guard;
  for (unsigned i = 0; i < devices_count; i++) {
    HANDLE_CUDA_ERROR(guard.set(device_ids[i]));
    HANDLE_CUDA_ERROR(bit_reverse_single(values, results, log_count, i, streams[i], log_devices_count));
  }
  HANDLE_CUDA_ERROR(guard.reset());
  HANDLE_CUDA_ERROR(sync_device_streams(device_ids, streams, log_devices_count));
  return cudaSuccess;
}

template cudaError_t bit_reverse_multigpu(const fd_q::storage **values, fd_q::storage **results, unsigned log_count, const cudaStream_t *streams,
                                          const int *device_ids, unsigned log_devices_count);

cudaError_t sort_u32(const sort_u32_configuration &configuration) {
  allocation<void> temp_storage;
  size_t temp_storage_bytes = 0;
  HANDLE_CUDA_ERROR(
      common::sort_keys(temp_storage, temp_storage_bytes, configuration.values, configuration.sorted_values, configuration.count, 0, 32, configuration.stream));
  HANDLE_CUDA_ERROR(allocate(temp_storage, temp_storage_bytes, configuration.mem_pool, configuration.stream));
  HANDLE_CUDA_ERROR(
      common::sort_keys(temp_storage, temp_storage_bytes, configuration.values, configuration.sorted_values, configuration.count, 0, 32, configuration.stream));
  HANDLE_CUDA_ERROR(free(temp_storage, configuration.stream));
  return cudaSuccess;
}

cudaError_t tear_down() {
  HANDLE_CUDA_ERROR(tear_down_powers_of_w());
  HANDLE_CUDA_ERROR(tear_down_powers_of_g_f());
  HANDLE_CUDA_ERROR(tear_down_powers_of_g_i());
  return cudaSuccess;
}

} // namespace ff
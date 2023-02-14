#include "bc.cuh"
#include "bellman-cuda.h"
#include "ff.cuh"
#include "msm.cuh"
#include "ntt.cuh"
#include "pn.cuh"
#include <cuda_runtime_api.h>

bc_error bc_get_device_count(int *count) { return static_cast<bc_error>(cudaGetDeviceCount(count)); }

bc_error bc_get_device(int *device_id) { return static_cast<bc_error>(cudaGetDevice(device_id)); }

bc_error bc_set_device(int device_id) { return static_cast<bc_error>(cudaSetDevice(device_id)); }

bc_error bc_stream_create(bc_stream *stream, bool blocking_sync) {
  return static_cast<bc_error>(bc::stream_create(reinterpret_cast<cudaStream_t &>(stream->handle), blocking_sync));
}

bc_error bc_stream_wait_event(bc_stream stream, bc_event event) {
  return static_cast<bc_error>(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream.handle), static_cast<cudaEvent_t>(event.handle)));
}

bc_error bc_stream_synchronize(bc_stream stream) { return static_cast<bc_error>(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.handle))); }

bc_error bc_stream_query(bc_stream stream) { return static_cast<bc_error>(cudaStreamQuery(static_cast<cudaStream_t>(stream.handle))); }

bc_error bc_stream_destroy(bc_stream stream) { return static_cast<bc_error>(cudaStreamDestroy(static_cast<cudaStream_t>(stream.handle))); }

bc_error bc_launch_host_fn(bc_stream stream, bc_host_fn fn, void *user_data) {
  return static_cast<bc_error>(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream.handle), fn, user_data));
}

bc_error bc_event_create(bc_event *event, bool blocking_sync, bool disable_timing) {
  int flags = (blocking_sync ? cudaEventBlockingSync : cudaEventDefault) | (disable_timing ? cudaEventDisableTiming : cudaEventDefault);
  return static_cast<bc_error>(cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&(event->handle)), flags));
}

bc_error bc_event_record(bc_event event, bc_stream stream) {
  return static_cast<bc_error>(cudaEventRecord(static_cast<cudaEvent_t>(event.handle), static_cast<cudaStream_t>(stream.handle)));
}

bc_error bc_event_synchronize(bc_event event) { return static_cast<bc_error>(cudaEventSynchronize(static_cast<cudaEvent_t>(event.handle))); }

bc_error bc_event_query(bc_event event) { return static_cast<bc_error>(cudaEventQuery(static_cast<cudaEvent_t>(event.handle))); }

bc_error bc_event_destroy(bc_event event) { return static_cast<bc_error>(cudaEventDestroy(static_cast<cudaEvent_t>(event.handle))); }

bc_error bc_event_elapsed_time(float *ms, bc_event start, bc_event end) {
  return static_cast<bc_error>(cudaEventElapsedTime(ms, static_cast<cudaEvent_t>(start.handle), static_cast<cudaEvent_t>(end.handle)));
}

bc_error bc_mem_get_info(size_t *free, size_t *total) { return static_cast<bc_error>(cudaMemGetInfo(free, total)); }

bc_error bc_malloc(void **ptr, size_t size) { return static_cast<bc_error>(cudaMalloc(ptr, size)); }

bc_error bc_malloc_host(void **ptr, size_t size) { return static_cast<bc_error>(cudaMallocHost(ptr, size)); }

bc_error bc_free(void *ptr) { return static_cast<bc_error>(cudaFree(ptr)); }

bc_error bc_free_host(void *ptr) { return static_cast<bc_error>(cudaFreeHost(ptr)); }

bc_error bc_host_register(void *ptr, size_t size) { return static_cast<bc_error>(cudaHostRegister(ptr, size, cudaHostRegisterDefault)); }

bc_error bc_host_unregister(void *ptr) { return static_cast<bc_error>(cudaHostUnregister(ptr)); }

bc_error bc_device_disable_peer_access(const int device_id) { return static_cast<bc_error>(cudaDeviceDisablePeerAccess(device_id)); }

bc_error bc_device_enable_peer_access(const int device_id) { return static_cast<bc_error>(cudaDeviceEnablePeerAccess(device_id, 0)); }

bc_error bc_memcpy(void *dst, const void *src, size_t count) { return static_cast<bc_error>(cudaMemcpy(dst, src, count, cudaMemcpyDefault)); }

bc_error bc_memcpy_async(void *dst, const void *src, size_t count, bc_stream stream) {
  return static_cast<bc_error>(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, static_cast<cudaStream_t>(stream.handle)));
}

bc_error bc_memset(void *ptr, int value, size_t count) { return static_cast<bc_error>(cudaMemset(ptr, value, count)); }

bc_error bc_memset_async(void *ptr, int value, size_t count, bc_stream stream) {
  return static_cast<bc_error>(cudaMemsetAsync(ptr, value, count, static_cast<cudaStream_t>(stream.handle)));
}

bc_error bc_mem_pool_create(bc_mem_pool *pool, int device_id) {
  return static_cast<bc_error>(bc::mem_pool_create(reinterpret_cast<cudaMemPool_t &>(pool->handle), device_id));
}

bc_error bc_mem_pool_destroy(bc_mem_pool pool) { return static_cast<bc_error>(cudaMemPoolDestroy(reinterpret_cast<cudaMemPool_t>(pool.handle))); }

bc_error bc_mem_pool_disable_peer_access(const bc_mem_pool pool, const int device_id) {
  cudaMemAccessDesc desc = {{cudaMemLocationTypeDevice, device_id}, cudaMemAccessFlagsProtNone};
  return static_cast<bc_error>(cudaMemPoolSetAccess(reinterpret_cast<cudaMemPool_t>(pool.handle), &desc, 1));
}

bc_error bc_mem_pool_enable_peer_access(const bc_mem_pool pool, const int device_id) {
  cudaMemAccessDesc desc = {{cudaMemLocationTypeDevice, device_id}, cudaMemAccessFlagsProtReadWrite};
  return static_cast<bc_error>(cudaMemPoolSetAccess(reinterpret_cast<cudaMemPool_t>(pool.handle), &desc, 1));
}

bc_error bc_malloc_from_pool_async(void **ptr, size_t size, bc_mem_pool pool, bc_stream stream) {
  return static_cast<bc_error>(cudaMallocFromPoolAsync(ptr, size, static_cast<cudaMemPool_t>(pool.handle), static_cast<cudaStream_t>(stream.handle)));
}

bc_error bc_free_async(void *ptr, bc_stream stream) { return static_cast<bc_error>(cudaFreeAsync(ptr, static_cast<cudaStream_t>(stream.handle))); }

bc_error ff_set_up(const unsigned powers_of_w_coarse_log_count, const unsigned powers_of_g_coarse_log_count) {
  return static_cast<bc_error>(ff::set_up(powers_of_w_coarse_log_count, powers_of_g_coarse_log_count));
}

bc_error ff_set_value(void *target, const void *value, unsigned count, bc_stream stream) {
  auto t = static_cast<fd_q::storage *>(target);
  auto v = static_cast<const fd_q::storage *>(value);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::set_value(t, v, count, s));
}
bc_error ff_set_value_zero(void *target, unsigned count, bc_stream stream) {
  auto t = static_cast<fd_q::storage *>(target);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::set_value_zero(t, count, s));
}

bc_error ff_set_value_one(void *target, unsigned count, bc_stream stream) {
  auto t = static_cast<fd_q::storage *>(target);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::set_value_one<fd_q>(t, count, s));
}

bc_error ff_ax(const void *a, const void *x, void *result, unsigned count, bc_stream stream) {
  auto as = static_cast<const fd_q::storage *>(a);
  auto xs = static_cast<const fd_q::storage *>(x);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::ax<fd_q>(as, xs, rs, count, s));
}

bc_error ff_a_plus_x(const void *a, const void *x, void *result, unsigned count, bc_stream stream) {
  auto as = static_cast<const fd_q::storage *>(a);
  auto xs = static_cast<const fd_q::storage *>(x);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::a_plus_x<fd_q>(as, xs, rs, count, s));
}

bc_error ff_x_plus_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::x_plus_y<fd_q>(xs, ys, rs, count, s));
}

bc_error ff_ax_plus_y(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto as = static_cast<const fd_q::storage *>(a);
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::ax_plus_y<fd_q>(as, xs, ys, rs, count, s));
}

bc_error ff_x_minus_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::x_minus_y<fd_q>(xs, ys, rs, count, s));
}

bc_error ff_ax_minus_y(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto as = static_cast<const fd_q::storage *>(a);
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::ax_minus_y<fd_q>(as, xs, ys, rs, count, s));
}

bc_error ff_x_minus_ay(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto as = static_cast<const fd_q::storage *>(a);
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::x_minus_ay<fd_q>(as, xs, ys, rs, count, s));
}

bc_error ff_x_mul_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream) {
  auto xs = static_cast<const fd_q::storage *>(x);
  auto ys = static_cast<const fd_q::storage *>(y);
  auto rs = static_cast<fd_q::storage *>(result);
  auto s = static_cast<cudaStream_t>(stream.handle);
  return static_cast<bc_error>(ff::x_mul_y<fd_q>(xs, ys, rs, count, s));
}

bc_error ff_grand_product(const ff_grand_product_configuration configuration) {
  ff::grand_product_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle), static_cast<cudaStream_t>(configuration.stream.handle),
                                         static_cast<fd_q::storage *>(configuration.inputs), static_cast<fd_q::storage *>(configuration.outputs),
                                         configuration.count};
  return static_cast<bc_error>(ff::grand_product(cfg));
}

bc_error ff_multiply_by_powers(ff_multiply_by_powers_configuration configuration) {
  ff::multiply_by_powers_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle), static_cast<cudaStream_t>(configuration.stream.handle),
                                              static_cast<fd_q::storage *>(configuration.inputs),        static_cast<fd_q::storage *>(configuration.base),
                                              static_cast<fd_q::storage *>(configuration.outputs),       configuration.count};
  return static_cast<bc_error>(ff::multiply_by_powers(cfg));
}

bc_error ff_inverse(const ff_inverse_configuration configuration) {
  ff::inverse_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle), static_cast<cudaStream_t>(configuration.stream.handle),
                                   static_cast<fd_q::storage *>(configuration.inputs), static_cast<fd_q::storage *>(configuration.outputs),
                                   configuration.count};
  return static_cast<bc_error>(ff::inverse(cfg));
}

bc_error ff_poly_evaluate(ff_poly_evaluate_configuration configuration) {
  ff::poly_evaluate_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle), static_cast<cudaStream_t>(configuration.stream.handle),
                                         static_cast<fd_q::storage *>(configuration.values),        static_cast<fd_q::storage *>(configuration.point),
                                         static_cast<fd_q::storage *>(configuration.result),        configuration.count};
  return static_cast<bc_error>(ff::poly_evaluate(cfg));
}

bc_error ff_get_powers_of_w(void *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse, const bool bit_reversed,
                            bc_stream stream) {
  return static_cast<bc_error>(ff::get_powers_of_w<fd_q>(static_cast<fd_q::storage *>(target), log_degree, offset, count, inverse, bit_reversed,
                                                         static_cast<cudaStream_t>(stream.handle)));
}

bc_error ff_get_powers_of_g(void *target, const unsigned log_degree, const unsigned offset, const unsigned count, const bool inverse, const bool bit_reversed,
                            bc_stream stream) {
  return static_cast<bc_error>(ff::get_powers_of_g<fd_q>(static_cast<fd_q::storage *>(target), log_degree, offset, count, inverse, bit_reversed,
                                                         static_cast<cudaStream_t>(stream.handle)));
}

bc_error ff_omega_shift(const void *values, void *result, const unsigned log_degree, const unsigned shift, const unsigned offset, const unsigned count,
                        const bool inverse, bc_stream stream) {
  return static_cast<bc_error>(ff::omega_shift<fd_q>(static_cast<const fd_q::storage *>(values), static_cast<fd_q::storage *>(result), log_degree, shift,
                                                     offset, count, inverse, static_cast<cudaStream_t>(stream.handle)));
}

bc_error ff_bit_reverse(const void *values, void *result, const unsigned log_count, bc_stream stream) {
  return static_cast<bc_error>(ff::bit_reverse<fd_q::storage>(static_cast<const fd_q::storage *>(values), static_cast<fd_q::storage *>(result), log_count,
                                                              static_cast<cudaStream_t>(stream.handle)));
}

bc_error ff_bit_reverse_multigpu(const void **values, void **results, const unsigned log_count, const bc_stream *streams, const int *device_ids,
                                 const unsigned log_devices_count) {
  return static_cast<bc_error>(ff::bit_reverse_multigpu<fd_q::storage>(reinterpret_cast<const fd_q::storage **>(values),
                                                                       reinterpret_cast<fd_q::storage **>(results), log_count,
                                                                       reinterpret_cast<const cudaStream_t *>(streams), device_ids, log_devices_count));
}

bc_error ff_select(const void *source, void *destination, const unsigned *indexes, const unsigned count, bc_stream stream) {
  return static_cast<bc_error>(ff::select<fd_q::storage>(static_cast<const fd_q::storage *>(source), static_cast<fd_q::storage *>(destination), indexes, count,
                                                         static_cast<cudaStream_t>(stream.handle)));
}

bc_error ff_sort_u32(const ff_sort_u32_configuration configuration) {
  const ff::sort_u32_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle), static_cast<cudaStream_t>(configuration.stream.handle),
                                          static_cast<unsigned int *>(configuration.values), static_cast<unsigned int *>(configuration.sorted_values),
                                          configuration.count};
  return static_cast<bc_error>(ff::sort_u32(cfg));
}

bc_error ff_tear_down() { return static_cast<bc_error>(ff::tear_down()); };

bc_error pn_set_up() { return static_cast<bc_error>(pn::set_up()); };

bc_error pn_generate_permutation_polynomials(generate_permutation_polynomials_configuration configuration) {
  pn::generate_permutation_polynomials_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                                            static_cast<cudaStream_t>(configuration.stream.handle),
                                                            configuration.indexes,
                                                            static_cast<fd_q::storage *>(configuration.scalars),
                                                            static_cast<fd_q::storage *>(configuration.target),
                                                            configuration.columns_count,
                                                            configuration.log_rows_count};
  return static_cast<bc_error>(pn::generate_permutation_polynomials(cfg));
}

bc_error pn_set_values_from_packed_bits(void *values, const void *packet_bits, const unsigned count, bc_stream stream) {
  return static_cast<bc_error>(pn::set_values_from_packed_bits(static_cast<fd_q::storage *>(values), static_cast<const unsigned *>(packet_bits), count,
                                                               static_cast<cudaStream_t>(stream.handle)));
}

bc_error pn_tear_down() { return static_cast<bc_error>(pn::tear_down()); };

bc_error msm_set_up() { return static_cast<bc_error>(msm::set_up()); }

bc_error msm_execute_async(const msm_configuration configuration) {
  msm::execution_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                      static_cast<cudaStream_t>(configuration.stream.handle),
                                      static_cast<msm::point_affine *>(configuration.bases),
                                      static_cast<fd_q::storage *>(configuration.scalars),
                                      static_cast<msm::point_jacobian *>(configuration.results),
                                      configuration.log_scalars_count,
                                      static_cast<cudaEvent_t>(configuration.h2d_copy_finished.handle),
                                      configuration.h2d_copy_finished_callback,
                                      configuration.h2d_copy_finished_callback_data,
                                      static_cast<cudaEvent_t>(configuration.d2h_copy_finished.handle),
                                      configuration.d2h_copy_finished_callback,
                                      configuration.d2h_copy_finished_callback_data};
  return static_cast<bc_error>(msm::execute_async(cfg));
}

bc_error msm_tear_down() { return static_cast<bc_error>(msm::tear_down()); };

bc_error ntt_set_up() { return static_cast<bc_error>(ntt::set_up()); }

bc_error ntt_execute_async(const ntt_configuration configuration) {
  ntt::execution_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                      static_cast<cudaStream_t>(configuration.stream.handle),
                                      static_cast<fd_q::storage *>(configuration.inputs),
                                      static_cast<fd_q::storage *>(configuration.outputs),
                                      configuration.log_values_count,
                                      configuration.bit_reversed_inputs,
                                      configuration.inverse,
                                      configuration.can_overwrite_inputs,
                                      configuration.log_extension_degree,
                                      configuration.coset_index,
                                      static_cast<cudaEvent_t>(configuration.h2d_copy_finished.handle),
                                      configuration.h2d_copy_finished_callback,
                                      configuration.h2d_copy_finished_callback_data,
                                      static_cast<cudaEvent_t>(configuration.d2h_copy_finished.handle),
                                      configuration.d2h_copy_finished_callback,
                                      configuration.d2h_copy_finished_callback_data};
  int dev;
  auto err = cudaGetDevice(&dev);
  if (err != cudaSuccess)
    return static_cast<bc_error>(err);
  return static_cast<bc_error>(ntt::execute_async_multigpu(&cfg, &dev, 0));
}

bc_error ntt_execute_async_multigpu(const ntt_configuration *configurations, const int *dev_ids, const unsigned log_n_devs) {
  int n_devs = 1 << log_n_devs;
  ntt::execution_configuration cfgs[n_devs];
  for (int i = 0; i < n_devs; i++) {
    const auto &configuration = configurations[i];
    cfgs[i] = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
               static_cast<cudaStream_t>(configuration.stream.handle),
               static_cast<fd_q::storage *>(configuration.inputs),
               static_cast<fd_q::storage *>(configuration.outputs),
               configuration.log_values_count,
               configuration.bit_reversed_inputs,
               configuration.inverse,
               configuration.can_overwrite_inputs,
               configuration.log_extension_degree,
               configuration.coset_index,
               static_cast<cudaEvent_t>(configuration.h2d_copy_finished.handle),
               configuration.h2d_copy_finished_callback,
               configuration.h2d_copy_finished_callback_data,
               static_cast<cudaEvent_t>(configuration.d2h_copy_finished.handle),
               configuration.d2h_copy_finished_callback,
               configuration.d2h_copy_finished_callback_data};
  }
  return static_cast<bc_error>(ntt::execute_async_multigpu(cfgs, dev_ids, log_n_devs));
}

bc_error ntt_tear_down() { return static_cast<bc_error>(ntt::tear_down()); };

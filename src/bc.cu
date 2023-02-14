#include "bc.cuh"
#include "common.cuh"
#include <cstdint>

namespace bc {

cudaError stream_create(cudaStream_t &stream, bool blocking_sync) {
  cudaError_t error = cudaStreamCreate(&stream);
  if (error != cudaSuccess || !blocking_sync)
    return error;
  cudaStreamAttrValue policy{};
  policy.syncPolicy = cudaSyncPolicyBlockingSync;
  return cudaStreamSetAttribute(stream, cudaStreamAttributeSynchronizationPolicy, &policy);
}

cudaError mem_pool_create(cudaMemPool_t &mem_pool, int device_id) {
  const cudaMemPoolProps props = {cudaMemAllocationTypePinned, cudaMemHandleTypeNone, {cudaMemLocationTypeDevice, device_id}};
  HANDLE_CUDA_ERROR(cudaMemPoolCreate(&mem_pool, &props));
  uint64_t mem_pool_threshold = UINT64_MAX;
  return cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &mem_pool_threshold);
}

} // namespace bc
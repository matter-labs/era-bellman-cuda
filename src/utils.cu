#include "common.cuh"
#include "utils.cuh"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
cudaError_t sync_device_streams(const int *const device_ids, const cudaStream_t *streams, const unsigned log_devices_count) {
  if (log_devices_count == 0)
    return cudaSuccess;
  const unsigned log_offset = log_devices_count - 1;
  const unsigned offset = 1 << log_offset;
  HANDLE_CUDA_ERROR(sync_device_streams(device_ids, streams, log_offset));
  HANDLE_CUDA_ERROR(sync_device_streams(device_ids + offset, streams + offset, log_offset));
  device_guard guard;
  for (unsigned i = 0; i < offset; i++) {
    unsigned indexes[] = {i, i + offset};
    for (unsigned j = 0; j < 2; j++) {
      cudaEvent_t event;
      HANDLE_CUDA_ERROR(guard.set(device_ids[indexes[j]]));
      HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      HANDLE_CUDA_ERROR(cudaEventRecord(event, streams[indexes[j]]));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(streams[indexes[1 - j]], event));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event));
    }
  }
  HANDLE_CUDA_ERROR(guard.reset());
  return cudaSuccess;
}
#pragma clang diagnostic pop

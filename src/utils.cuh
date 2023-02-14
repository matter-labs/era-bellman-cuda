#pragma once
#include "cuda_runtime_api.h"

struct device_guard {
  cudaError_t set(const int device_id) {
    if (previous_device_id == -1)
      HANDLE_CUDA_ERROR(cudaGetDevice(&previous_device_id));
    return cudaSetDevice(device_id);
  }

  cudaError_t reset() {
    if (previous_device_id == -1)
      return cudaSuccess;
    auto result = cudaSetDevice(previous_device_id);
    previous_device_id = -1;
    return result;
  }

  ~device_guard() { reset(); }

private:
  int previous_device_id = -1;
};

cudaError_t sync_device_streams(const int *device_ids, const cudaStream_t *streams, unsigned log_devices_count);

#pragma once

#include "bellman-cuda.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand.h>
#include <gtest/gtest.h>

#define ASSERT_BC_SUCCESS(error) ASSERT_EQ(error, bc_success)

#define ASSERT_CUDA_SUCCESS(error) ASSERT_EQ(error, cudaSuccess)

#define ASSERT_CURAND_SUCCESS(error) ASSERT_EQ(error, CURAND_STATUS_SUCCESS)

template <class FD> void fields_populate_random_host(typename FD::storage *fields, unsigned count, uint64_t seed);

template <class FD> void fields_populate_random_host(typename FD::storage *fields, unsigned count);

curandStatus_t populate_random_device(unsigned *values, size_t count, uint64_t seed);

curandStatus_t populate_random_device(unsigned *values, size_t count);

cudaError_t trim_to_mask(unsigned *values, unsigned mask, unsigned count);

template <class FD> int fields_populate_random_device(typename FD::storage *fields, unsigned count, uint64_t seed);

template <class FD> int fields_populate_random_device(typename FD::storage *fields, unsigned count);

template <class FD> cudaError_t fields_set(typename FD::storage *fields, const typename FD::storage &value, unsigned count);

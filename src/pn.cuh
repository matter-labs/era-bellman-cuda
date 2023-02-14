#pragma once

#include "pn_kernels.cuh"

namespace pn {

struct generate_permutation_polynomials_configuration {
  cudaMemPool_t mem_pool;
  cudaStream_t stream;
  unsigned *indexes;
  fd_q::storage *scalars;
  fd_q::storage *target;
  unsigned columns_count;
  unsigned log_rows_count;
};

cudaError_t set_up();

cudaError_t generate_permutation_polynomials(const generate_permutation_polynomials_configuration &cfg);

cudaError_t tear_down();

} // namespace pn

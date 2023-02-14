#include "common.cuh"
#include "ff_dispatch_st.cuh"
#include "memory.cuh"
#include <random>

template <class FD> void fields_populate_random_host(typename FD::storage *fields, const unsigned count, const uint64_t seed) {
  typedef typename FD::storage storage;
  constexpr unsigned limbs_count = FD::TLC;
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned> distribution;
  for (unsigned i = 0; i < count; i++) {
    storage value{};
    for (unsigned j = 0; j < limbs_count; j++)
      value.limbs[j] = distribution(generator);
    value.limbs[limbs_count - 1] &= 0x3fffffffu;
    fields[i] = FD::reduce(value);
  }
}

template <class FD> void fields_populate_random_host(typename FD::storage *fields, const unsigned count) {
  std::random_device rd;
  fields_populate_random_host<FD>(fields, count, rd());
}

template <class FD> __global__ void trim_to_modulus(typename FD::storage *x, const unsigned n) {
  typedef typename FD::storage storage;
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n)
    return;
  storage value = memory::load(x + gid);
  value.limbs[FD::TLC - 1] &= 0x3fffffffu;
  value = FD::reduce(value);
  memory::store(x + gid, value);
}

curandStatus_t populate_random_device(unsigned *values, const size_t count, const uint64_t seed) {
  curandGenerator_t gen;
  curandStatus_t status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  if (status != CURAND_STATUS_SUCCESS)
    return status;
  status = curandSetPseudoRandomGeneratorSeed(gen, seed);
  if (status != CURAND_STATUS_SUCCESS)
    return status;
  return curandGenerate(gen, values, count);
}

template <class FD> int fields_populate_random_device(typename FD::storage *fields, const unsigned count, const uint64_t seed) {
  constexpr unsigned limbs_count = FD::TLC;
  curandStatus_t status = populate_random_device(reinterpret_cast<unsigned *>(fields), count * limbs_count, seed);
  if (status != CURAND_STATUS_SUCCESS)
    return status;
  unsigned blocks_count = ((count - 1) / 32) + 1;
  trim_to_modulus<FD><<<blocks_count, 32>>>(fields, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

curandStatus_t populate_random_device(unsigned *values, const size_t count) {
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> dist;
  auto seed = dist(eng);
  return populate_random_device(values, count, seed);
}

__global__ void trim_to_mask_kernel(unsigned *values, const unsigned mask, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  values[gid] &= mask;
}

cudaError_t trim_to_mask(unsigned *values, const unsigned mask, const unsigned count) {
  unsigned blocks_count = ((count - 1) / 32) + 1;
  trim_to_mask_kernel<<<blocks_count, 32>>>(values, mask, count);
  return cudaGetLastError();
}

template <class FD> int fields_populate_random_device(typename FD::storage *fields, const unsigned count) {
  std::random_device rd;
  std::mt19937_64 eng(rd());
  std::uniform_int_distribution<unsigned long long> dist;
  auto seed = dist(eng);
  return fields_populate_random_device<FD>(fields, count, seed);
}

template <class FD> __global__ void fields_set_kernel(typename FD::storage *fields, typename FD::storage value, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  fields[gid] = value;
}

template <class FD> cudaError_t fields_set(typename FD::storage *fields, const typename FD::storage &value, const unsigned count) {
  fields_set_kernel<FD><<<(count - 1) / 32 + 1, 32>>>(fields, value, count);
  cudaError_t error = cudaGetLastError();
  return error ? error : cudaDeviceSynchronize();
}

// forcing instantiation of no-seed-arg overloads should be enough, because they call seed-arg overloads.
template void fields_populate_random_host<fd_p>(fd_p::storage *, unsigned);

template void fields_populate_random_host<fd_q>(fd_q::storage *, unsigned);

template int fields_populate_random_device<fd_p>(fd_p::storage *, unsigned);

template int fields_populate_random_device<fd_q>(fd_q::storage *, unsigned);

template cudaError_t fields_set<fd_p>(fd_p::storage *, const fd_p::storage &, unsigned);

template cudaError_t fields_set<fd_q>(fd_q::storage *, const fd_q::storage &, unsigned);

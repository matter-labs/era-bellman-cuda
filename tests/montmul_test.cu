#include "bc.cuh"
#include "common.cuh"
#include "msm.cuh"
#include "msm_bases.cuh"
#include <algorithm>
#include <curand.h>
#include <numeric>
#include <random>

typedef fd_p fp;
typedef fd_q fq;

constexpr unsigned LOG_NUMEL = 26;
constexpr unsigned LOG_NTHREADS = 8;
constexpr unsigned NTHREADS = 1 << LOG_NTHREADS;
constexpr unsigned PERF_REPS = 16;

template <typename FD, bool PLUS, unsigned INPUT_RANGE>
__launch_bounds__(256, 1) __global__
    void reduce_ab_plusorminus_cd_correctness_kernel(const typename FD::storage *a, const typename FD::storage *b, const typename FD::storage *c,
                                                     const typename FD::storage *d, const unsigned count) {
  typedef typename FD::storage storage;
  typedef typename FD::storage_wide storage_wide;
  constexpr storage modulus = FD::CONFIG::modulus;

  const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid > count)
    return;

  auto as = memory::load<storage>(a + tid);
  auto bs = memory::load<storage>(b + tid);
  auto cs = memory::load<storage>(c + tid);
  auto ds = memory::load<storage>(d + tid);
  // Forces some range corner cases
  if (tid == 0) {
    as = modulus;
    bs = modulus;
    cs = modulus;
    ds = modulus;
    as.limbs[0]--;
    bs.limbs[0]--;
    cs.limbs[0]--;
    ds.limbs[0]--;
  } else if (tid == 1) {
    as = {0};
    bs = {0};
    cs = modulus;
    ds = modulus;
    cs.limbs[0]--;
    ds.limbs[0]--;
  } else if (tid == 2) {
    as = modulus;
    bs = modulus;
    cs = {0};
    ds = {0};
    as.limbs[0]--;
    bs.limbs[0]--;
  }

  if (INPUT_RANGE == 2) {
    if (tid == 1) {
      cs = FD::add<0>(cs, modulus);
      ds = FD::add<0>(ds, modulus);
    } else if (tid == 2) {
      as = FD::add<0>(as, modulus);
      bs = FD::add<0>(bs, modulus);
    } else {
      as = FD::add<0>(as, modulus);
      bs = FD::add<0>(bs, modulus);
      cs = FD::add<0>(cs, modulus);
      ds = FD::add<0>(ds, modulus);
    }
  }

  auto abs = FD::mul(as, bs);
  auto cds = FD::mul(cs, ds);
  // abs and cds should both be in [0, mod) if INPUT_RANGE = 1 and [0, 2*mod) if INPUT_RANGE = 2
  auto control = PLUS ? FD::add(abs, cds) : FD::sub(abs, cds);

  if (PLUS && INPUT_RANGE == 2) {
    bs = FD::reduce<1>(bs);
    ds = FD::reduce<1>(ds);
  }
  storage_wide abs_wide = FD::mul_wide<0>(as, bs);
  storage_wide cds_wide = FD::mul_wide<0>(cs, ds);
  // if PLUS, the results of each mul will be in [0, 2*mod^2) even if DOUBLE_RANGE,
  // so the output will be in [0, 4*mod^2) which redc_wide can handle, and will
  // pull back into [0, 2*mod), so add_wide<0> is ok.
  // if not PLUS, we are doing [0, 4*mod) - [0, 4*mod), so we need sub_wide<4>
  // to ensure result is in [0, 4*mod) which, again, redc_wide can handle and will
  // pull back into [0, 2*mod).
  storage_wide ab_cds_wide = PLUS ? FD::add_wide<0>(abs_wide, cds_wide) : FD::sub_wide<4>(abs_wide, cds_wide);
  storage ab_cds = FD::redc_wide<0>(ab_cds_wide);

  // also test the inplace alternative
  FD::redc_wide_inplace(ab_cds_wide);
  storage ab_cds_from_inplace = ab_cds_wide.get_lo();

  assert(FD::eq(FD::reduce<1>(control), FD::reduce<1>(ab_cds)));
  assert(FD::eq(FD::reduce<1>(control), FD::reduce<1>(ab_cds_from_inplace)));
}

template <typename FD, bool PLUS>
__launch_bounds__(256, 1) __global__
    void reduce_ab_plusorminus_reduce_cd_perf_kernel(const typename FD::storage *a, const typename FD::storage *b, const typename FD::storage *c,
                                                     const typename FD::storage *d, typename FD::storage *r) {
  typedef typename FD::storage storage;

  // Every thread acts redundantly on the same memory. We don't care about results, we just care about
  // minimizing memory overhead while preventing nvcc from optimizing away any realistic compute work.
  for (int rep = 0; rep < PERF_REPS; rep++) {
    auto as = memory::load<storage>(a + rep);
    auto bs = memory::load<storage>(b + rep);
    auto cs = memory::load<storage>(c + rep);
    auto ds = memory::load<storage>(d + rep);

    auto abs = FD::mul(as, bs);
    auto cds = FD::mul(cs, ds);
    auto ab_cds = PLUS ? FD::add(abs, cds) : FD::sub(abs, cds);

    memory::store<storage>(r + rep, ab_cds);
  }
}

template <typename FD, bool PLUS, unsigned INPUT_RANGE>
__launch_bounds__(256, 1) __global__
    void reduce_ab_plusorminus_cd_perf_kernel(const typename FD::storage *a, const typename FD::storage *b, const typename FD::storage *c,
                                              const typename FD::storage *d, typename FD::storage *r) {
  typedef typename FD::storage storage;
  typedef typename FD::storage_wide storage_wide;

  // Every thread acts redundantly on the same memory. We don't care about results, we just care about
  // minimizing memory overhead while preventing nvcc from optimizing away any realistic compute work.
  for (int rep = 0; rep < PERF_REPS; rep++) {
    auto as = memory::load<storage>(a + rep);
    auto bs = memory::load<storage>(b + rep);
    auto cs = memory::load<storage>(c + rep);
    auto ds = memory::load<storage>(d + rep);

    if (PLUS && INPUT_RANGE == 2) {
      bs = FD::reduce<1>(bs);
      ds = FD::reduce<1>(ds);
    }
    storage_wide abs_wide = FD::mul_wide<0>(as, bs);
    storage_wide cds_wide = FD::mul_wide<0>(cs, ds);
    storage_wide ab_cds_wide = PLUS ? FD::add_wide<0>(abs_wide, cds_wide) : FD::sub_wide<4>(abs_wide, cds_wide);
    // storage ab_cds = FD::redc_wide(ab_cds_wide);
    FD::redc_wide_inplace(ab_cds_wide);
    storage ab_cds_from_inplace = ab_cds_wide.get_lo();
    ab_cds_from_inplace = FD::reduce<1>(ab_cds_from_inplace);

    memory::store<storage>(r + rep, ab_cds_from_inplace);
  }
}

class montmul_test : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_CUDA_SUCCESS(cudaDeviceReset()); }
  void TearDown() override { cudaDeviceReset(); }

  cudaMemPool_t pool;
  cudaStream_t stream;

  template <typename FD> void set_up(typename FD::storage ***arrays, bool *arrays_need_fill, unsigned arrays_count, unsigned numel) {
    typedef typename FD::storage storage;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    for (int i = 0; i < arrays_count; i++) {
      ASSERT_CUDA_SUCCESS(cudaMallocAsync(arrays[i], sizeof(storage) * numel, pool, stream));
      if (arrays_need_fill[i]) {
        ASSERT_CUDA_SUCCESS(fields_populate_random_device<FD>(*arrays[i], numel));
      }
    }
  }

  template <typename FD> void tear_down(typename FD::storage ***arrays, unsigned arrays_count) {
    for (int i = 0; i < arrays_count; i++) {
      cudaFreeAsync(*arrays[i], stream);
    }
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  }

  template <typename FD, bool PLUS, unsigned INPUT_RANGE> void reduce_ab_plusorminus_cd_correctness(const unsigned log_count) {
    typedef typename FD::storage storage;
    constexpr unsigned arrays_count = 4;
    const unsigned numel = 1 << log_count;
    storage *a, *b, *c, *d;
    storage **arrays[4] = {&a, &b, &c, &d};
    bool arrays_need_fill[4] = {1, 1, 1, 1};
    set_up<FD>(arrays, arrays_need_fill, arrays_count, numel);
    const unsigned nblocks = (numel + NTHREADS - 1) / NTHREADS;
    reduce_ab_plusorminus_cd_correctness_kernel<FD, PLUS, INPUT_RANGE><<<nblocks, NTHREADS, 0, stream>>>(a, b, c, d, numel);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    tear_down<FD>(arrays, arrays_count);
  }

  template <typename FD> void reduce_ab_plusorminus_cd_benchmark(const unsigned log_count) {
    typedef typename FD::storage storage;
    constexpr unsigned arrays_count = 5;
    storage *a, *b, *c, *d, *r;
    storage **arrays[5] = {&a, &b, &c, &d, &r};
    bool arrays_need_fill[5] = {1, 1, 1, 1, 0};
    set_up<FD>(arrays, arrays_need_fill, arrays_count, PERF_REPS);
    cudaEvent_t start, end;
    float elapsed;
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
    printf("\nLimb count: %d\n", FD::TLC);
    const unsigned numel = 1 << log_count;
    const unsigned nblocks = numel >> LOG_NTHREADS;

    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    reduce_ab_plusorminus_reduce_cd_perf_kernel<FD, true><<<nblocks, NTHREADS, 0, stream>>>(a, b, c, d, r);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("reduce(a*b) + reduce(c*d)      on %zd (ab, cd) pairs took %8.3f ms\n", size_t(numel) * PERF_REPS, elapsed);

    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    reduce_ab_plusorminus_cd_perf_kernel<FD, true, 1><<<nblocks, NTHREADS, 0, stream>>>(a, b, c, d, r);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("reduce(a*b + c*d)              on %zd (ab, cd) pairs took %8.3f ms\n", size_t(numel) * PERF_REPS, elapsed);

    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    reduce_ab_plusorminus_cd_perf_kernel<FD, true, 2><<<nblocks, NTHREADS, 0, stream>>>(a, b, c, d, r);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("reduce(a*b + c*d) double range on %zd (ab, cd) pairs took %8.3f ms\n", size_t(numel) * PERF_REPS, elapsed);

    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    reduce_ab_plusorminus_cd_perf_kernel<FD, false, 1><<<nblocks, NTHREADS, 0, stream>>>(a, b, c, d, r);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("reduce(a*b - c*d)              on %zd (ab, cd) pairs took %8.3f ms\n", size_t(numel) * PERF_REPS, elapsed);

    printf("\n");
    ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
    ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
    tear_down<FD>(arrays, arrays_count);
  }
};

TEST_F(montmul_test, reduce_ab_plus_cd_fp_correctness) { reduce_ab_plusorminus_cd_correctness<fp, true, 1>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_plus_cd_fq_correctness) { reduce_ab_plusorminus_cd_correctness<fq, true, 1>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_plus_cd_fp_correctness_double_range) { reduce_ab_plusorminus_cd_correctness<fp, true, 2>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_plus_cd_fq_correctness_double_range) { reduce_ab_plusorminus_cd_correctness<fq, true, 2>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_minus_cd_fp_correctness) { reduce_ab_plusorminus_cd_correctness<fp, false, 1>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_minus_cd_fq_correctness) { reduce_ab_plusorminus_cd_correctness<fq, false, 1>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_minus_cd_fp_correctness_double_range) { reduce_ab_plusorminus_cd_correctness<fp, false, 2>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_minus_cd_fq_correctness_double_range) { reduce_ab_plusorminus_cd_correctness<fq, false, 2>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_plus_cd_fp_benchmark) { reduce_ab_plusorminus_cd_benchmark<fp>(LOG_NUMEL); }

TEST_F(montmul_test, reduce_ab_plus_cd_fq_benchmark) { reduce_ab_plusorminus_cd_benchmark<fq>(LOG_NUMEL); }

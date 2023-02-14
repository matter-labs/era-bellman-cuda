#include "bc.cuh"
#include "common.cuh"
#include "msm.cuh"
#include "msm_bases.cuh"
#include <algorithm>
#include <curand.h>
#include <numeric>
#include <random>

using namespace msm;
using namespace std;
typedef fd_p fp;
typedef fd_q fq;
typedef curve::point_affine pa;
typedef curve::point_jacobian point;

__global__ void set_bases(pa *bases, unsigned *base_multipliers, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned base_index = base_multipliers[gid] & 0xff;
  const pa base = pa::to_montgomery(g_bases[base_index], fp());
  bases[gid] = base;
  base_multipliers[gid] = base_index + 1;
}

__global__ void multiply_scalars_kernel(fq::storage *scalars, const unsigned *base_multipliers, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  scalars[gid] = fq::from_montgomery(fq::mul(base_multipliers[gid], scalars[gid]));
}

class msm_test : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_CUDA_SUCCESS(cudaDeviceReset()); }
  void TearDown() override { cudaDeviceReset(); }

  void set_up(const unsigned log_max_bases_count) {
    ASSERT_CUDA_SUCCESS(msm::set_up());
    const unsigned max_bases_count = 1 << log_max_bases_count;
    base_multipliers = new unsigned[max_bases_count];
    u_bases = new pa[max_bases_count];
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_bases, sizeof(pa) * max_bases_count));
    u_scalars = new fq::storage[max_bases_count];
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_scalars, sizeof(fq::storage) * max_bases_count));
    ASSERT_CUDA_SUCCESS(cudaMallocHost(&h_scalars, sizeof(fq::storage) * max_bases_count));
    u_results = new point[256];
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_results, sizeof(point) * 256));
    ASSERT_CUDA_SUCCESS(cudaMallocHost(&h_results, sizeof(point) * 256));
  }

  void tear_down() {
    delete[] u_bases;
    delete[] base_multipliers;
    delete[] u_scalars;
    ASSERT_CUDA_SUCCESS(cudaFree(d_bases));
    ASSERT_CUDA_SUCCESS(cudaFree(d_scalars));
    ASSERT_CUDA_SUCCESS(cudaFreeHost(h_scalars));
    delete[] u_results;
    ASSERT_CUDA_SUCCESS(cudaFree(d_results));
    ASSERT_CUDA_SUCCESS(cudaFreeHost(h_results));
    ASSERT_CUDA_SUCCESS(msm::tear_down());
  }

  unsigned *base_multipliers{};
  pa *u_bases{};
  pa *d_bases{};
  fq::storage *u_scalars{};
  fq::storage *d_scalars{};
  fq::storage *h_scalars{};
  point *u_results{};
  point *d_results{};
  point *h_results{};

  void generate_bases(const unsigned log_count) {
    const unsigned count = 1 << log_count;
    curandGenerator_t gen;
    ASSERT_CURAND_SUCCESS(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    random_device rd;
    mt19937_64 eng(rd());
    uniform_int_distribution<unsigned long long> dist;
    auto seed = dist(eng);
    ASSERT_CURAND_SUCCESS(curandSetPseudoRandomGeneratorSeed(gen, seed));
    unsigned *d_base_multipliers;
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_base_multipliers, sizeof(unsigned) * count));
    ASSERT_CURAND_SUCCESS(curandGenerate(gen, d_base_multipliers, count));
    unsigned blocks_count = ((count - 1) / 32) + 1;
    set_bases<<<blocks_count, 32>>>(d_bases, d_base_multipliers, count);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaMemcpy(base_multipliers, d_base_multipliers, sizeof(unsigned) * count, cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(u_bases, d_bases, sizeof(pa) * count, cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaFree(d_base_multipliers));
  }

  void multiply_scalars(const unsigned count) {
    unsigned blocks_count = ((count - 1) / 32) + 1;
    unsigned *d_base_multipliers;
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_base_multipliers, sizeof(unsigned) * count));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(d_base_multipliers, base_multipliers, sizeof(unsigned) * count, cudaMemcpyHostToDevice));
    multiply_scalars_kernel<<<blocks_count, 32>>>(d_scalars, d_base_multipliers, count);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaFree(d_base_multipliers));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(u_scalars, d_scalars, sizeof(fq::storage) * count, cudaMemcpyDeviceToHost));
  }

  point compute_checksum(const unsigned count) {
    multiply_scalars(count);
    fq::storage sum{};
    for (unsigned i = 0; i < count; i++)
      sum = fq::add(sum, u_scalars[i]);
    fp d = fp();
    return curve::mul<fq>(sum, pa::to_jacobian(pa::to_montgomery(g_bases[0], d), d), d);
  }

  point compute_result(const unsigned log_count) {
    const unsigned result_bits_count = 254;
    fp d = fp();
    point sum = point::point_at_infinity(d);
    for (unsigned i = 0; i < result_bits_count; i++) {
      unsigned index = result_bits_count - i - 1;
      point bucket = u_results[index];
      sum = i == 0 ? bucket : curve::add(curve::dbl(sum, d), bucket, d);
    }
    return sum;
  }

  static bool point_eq(const point &p1, const point &p2) { return point::eq(p1, p2, fp()); }

  void generate_scalars(const unsigned log_count) const {
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<fq>(d_scalars, 1 << log_count));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(h_scalars, d_scalars, sizeof(fq::storage) << log_count, cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(u_scalars, h_scalars, sizeof(fq::storage) << log_count, cudaMemcpyHostToHost));
  }

  void verify_result(execution_configuration cfg) {
    if (cfg.results != u_results)
      ASSERT_CUDA_SUCCESS(cudaMemcpy(u_results, cfg.results, sizeof(point) * 256, cudaMemcpyDefault));
    point sum = compute_result(cfg.log_scalars_count);
    point checksum = compute_checksum(1 << cfg.log_scalars_count);
    ASSERT_PRED2(point_eq, sum, checksum);
  }

  static void preallocate_pool(cudaMemPool_t pool, const unsigned slack) {
    size_t mem_free = 0;
    size_t mem_total = 0;
    ASSERT_CUDA_SUCCESS(cudaMemGetInfo(&mem_free, &mem_total));
    void *dummy;
    size_t dummy_size = ((mem_free >> slack) - 1) << slack;
    ASSERT_CUDA_SUCCESS(cudaMallocFromPoolAsync(&dummy, dummy_size, pool, nullptr));
    ASSERT_CUDA_SUCCESS(cudaFreeAsync(dummy, nullptr));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  }

  void correctness(const unsigned log_count) {
    set_up(log_count);
    generate_bases(log_count);
    generate_scalars(log_count);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    execution_configuration cfg = {mem_pool, stream, d_bases, u_scalars, u_results, log_count};
    ASSERT_CUDA_SUCCESS(execute_async(cfg));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    verify_result(cfg);
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  void benchmark(const vector<unsigned> &log_counts, const vector<cudaMemoryType> &types) {
    const unsigned max_log_count = *max_element(log_counts.begin(), log_counts.end());
    set_up(max_log_count);
    generate_bases(max_log_count);
    generate_scalars(max_log_count);
    fq::storage *scalars;
    point *results;
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    preallocate_pool(mem_pool, 25);
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, d_bases, d_scalars, d_results, max_log_count}));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    printf("size");
    for (cudaMemoryType type : types) {
      string type_label;
      switch (type) {
      case cudaMemoryTypeUnregistered:
        type_label = "pageable";
        break;
      case cudaMemoryTypeHost:
        type_label = "pinned";
        break;
      case cudaMemoryTypeDevice:
        type_label = "device";
        break;
      default:
        FAIL();
      }
      printf("\t%11s", type_label.c_str());
    }
    printf("\n");
    for (unsigned log_count : log_counts) {
      printf("2^%d", log_count);
      for (cudaMemoryType type : types) {
        switch (type) {
        case cudaMemoryTypeUnregistered:
          scalars = u_scalars;
          results = u_results;
          break;
        case cudaMemoryTypeHost:
          scalars = h_scalars;
          results = h_results;
          break;
        case cudaMemoryTypeDevice:
          scalars = d_scalars;
          results = d_results;
          break;
        default:
          FAIL();
        }
        execution_configuration cfg = {mem_pool, stream, d_bases, scalars, results, log_count};
        cudaEvent_t start;
        cudaEvent_t end;
        ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
        ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
        ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
        ASSERT_CUDA_SUCCESS(execute_async(cfg));
        ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
        ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
        float elapsed;
        ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
        printf("\t%8.3f ms", elapsed);
        ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
        ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
      }
      printf("\n");
    }
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }
};

TEST_F(msm_test, correctness_minimum_size) { correctness(0); }

TEST_F(msm_test, correctness_size_10) { correctness(10); }

TEST_F(msm_test, correctness_size_20) { correctness(20); }

TEST_F(msm_test, benchmark_range) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 26;
  vector<unsigned> log_counts(max_log_count - min_log_count + 1);
  iota(log_counts.begin(), log_counts.end(), min_log_count);
  const vector<cudaMemoryType> types = {cudaMemoryTypeDevice, cudaMemoryTypeHost, cudaMemoryTypeUnregistered};
  benchmark(log_counts, types);
}

TEST_F(msm_test, memory_requirements) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 26;
  size_t zero = 0;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&d_bases, sizeof(pa) << max_log_count));
  ASSERT_CUDA_SUCCESS(fields_populate_random_device<fp>(reinterpret_cast<fp::storage *>(d_bases), 2 << max_log_count));
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  printf("size\t%11s\t%11s\t%11s\t%11s\t%11s\n", "16 loops", "8 loops", "4 loops", "2 loops", "1 loop");
  for (unsigned i = min_log_count; i <= max_log_count; i++) {
    printf("2^%2d", i);
    execution_configuration cfg = {pool, stream, d_bases, reinterpret_cast<fq::storage *>(d_bases), reinterpret_cast<point_jacobian *>(d_bases), i};
    cfg.force_max_chunk_size = true;
    cfg.log_max_chunk_size = i - 4;
    if (execute_async(cfg) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    cfg.log_max_chunk_size = i - 3;
    if (execute_async(cfg) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    cfg.log_max_chunk_size = i - 2;
    if (execute_async(cfg) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    cfg.log_max_chunk_size = i - 1;
    if (execute_async(cfg) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    cfg.log_max_chunk_size = i - 0;
    if (execute_async(cfg) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    printf("\n");
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  ASSERT_CUDA_SUCCESS(cudaFree(d_bases));
}

TEST_F(msm_test, benchmark_loops) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 26;
  set_up(max_log_count);
  generate_bases(max_log_count);
  generate_scalars(max_log_count);
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  preallocate_pool(pool, 25);
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  cudaEvent_t start;
  cudaEvent_t end;
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
  printf("size\t%11s\t%11s\t%11s\t%11s\t%11s\n", "16 loops", "8 loops", "4 loops", "2 loops", "1 loop");
  for (unsigned i = min_log_count; i <= max_log_count; i++) {
    printf("2^%2d", i);
    execution_configuration cfg = {pool, stream, d_bases, d_scalars, d_results, i};
    cfg.force_max_chunk_size = true;
    cfg.log_max_chunk_size = i - 4;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    if (execute_async(cfg) == cudaSuccess) {
      ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
      float elapsed;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
      printf("\t%8.3f ms", elapsed);
    } else
      printf("\t%11s", "N/A");
    cfg.log_max_chunk_size = i - 3;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    if (execute_async(cfg) == cudaSuccess) {
      ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
      float elapsed;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
      printf("\t%8.3f ms", elapsed);
    } else
      printf("\t%11s", "N/A");
    cfg.log_max_chunk_size = i - 2;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    if (execute_async(cfg) == cudaSuccess) {
      ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
      float elapsed;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
      printf("\t%8.3f ms", elapsed);
    } else
      printf("\t%11s", "N/A");
    cfg.log_max_chunk_size = i - 1;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    if (execute_async(cfg) == cudaSuccess) {
      ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
      float elapsed;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
      printf("\t%8.3f ms", elapsed);
    } else
      printf("\t%11s", "N/A");
    cfg.log_max_chunk_size = i - 0;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    if (execute_async(cfg) == cudaSuccess) {
      ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
      float elapsed;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
      printf("\t%8.3f ms", elapsed);
    } else
      printf("\t%11s", "N/A");
    printf("\n");
  }
  ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
  ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  ASSERT_CUDA_SUCCESS(cudaFree(d_bases));
}

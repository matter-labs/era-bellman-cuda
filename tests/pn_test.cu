#include "bc.cuh"
#include "common.cuh"
#include "ff.cuh"
#include "pn.cuh"

class pn_test : public ::testing::Test {
protected:
  typedef fd_q f;
  typedef f::storage s;
  static const unsigned log_n = 16;
  static const unsigned n = 1 << log_n;

  pn_test() {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&x, n * sizeof(s)));
    assert(!cudaMallocManaged(&y, n * sizeof(s)));
    assert(!cudaMallocManaged(&zeroes, n * sizeof(s)));
    assert(!cudaMallocManaged(&ones, n * sizeof(s)));
    assert(!cudaMallocManaged(&twos, n * sizeof(s)));
    assert(!cudaMallocManaged(&result1, n * sizeof(s)));
    assert(!cudaMallocManaged(&result2, n * sizeof(s)));
  }

  ~pn_test() override {
    cudaFree(x);
    cudaFree(y);
    cudaFree(zeroes);
    cudaFree(ones);
    cudaFree(twos);
    cudaFree(result1);
    cudaFree(result2);
    cudaDeviceReset();
  }

  void SetUp() override {
    ASSERT_CUDA_SUCCESS(ff::set_up(25, 14));
    ASSERT_CUDA_SUCCESS(pn::set_up());
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(x, n));
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(x, n));
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(y, n));
    ASSERT_CUDA_SUCCESS(cudaMemset(zeroes, 0, n * sizeof(s)));
    ASSERT_CUDA_SUCCESS(fields_set<f>(ones, f::get_one(), n));
    ASSERT_CUDA_SUCCESS(fields_set<f>(twos, f::dbl(f::get_one()), n));
    ASSERT_CUDA_SUCCESS(cudaMemset(result1, 0, n * sizeof(f)));
    ASSERT_CUDA_SUCCESS(cudaMemset(result2, 0, n * sizeof(f)));
  }

  void TearDown() override {
    ASSERT_CUDA_SUCCESS(pn::tear_down());
    ASSERT_CUDA_SUCCESS(ff::tear_down());
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

  s *x{};
  s *y{};
  s *zeroes{};
  s *ones{};
  s *twos{};
  s *result1{};
  s *result2{};
};

TEST_F(pn_test, transpose_correctness) {
  const unsigned col_count = 4;
  const unsigned log_row_count = 10;
  const unsigned cell_count = col_count << log_row_count;
  unsigned *v1;
  unsigned *v2;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v1, cell_count * sizeof(unsigned)));
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v2, cell_count * sizeof(unsigned)));
  for (unsigned i = 0; i < cell_count; i++)
    v1[i] = i;
  ASSERT_CUDA_SUCCESS(pn::transpose<col_count>(v2, v1, log_row_count, nullptr));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  for (unsigned i = 0; i < cell_count; i++) {
    const unsigned col = i % col_count;
    const unsigned row = i / col_count;
    const unsigned value = row + (col << log_row_count);
    ASSERT_EQ(v2[i], value);
  }
}

TEST_F(pn_test, transpose_transpose) {
  const unsigned col_count = 4;
  const unsigned log_row_count = 2;
  const unsigned cell_count = col_count << log_row_count;
  unsigned *v1;
  unsigned *v2;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v1, cell_count * sizeof(unsigned)));
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v2, cell_count * sizeof(unsigned)));
  for (unsigned i = 0; i < cell_count; i++)
    v1[i] = i;
  ASSERT_CUDA_SUCCESS(pn::transpose<col_count>(v2, v1, log_row_count, nullptr));
  ASSERT_CUDA_SUCCESS(pn::transpose<col_count>(v1, v2, log_row_count, nullptr));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  for (unsigned i = 0; i < cell_count; i++)
    ASSERT_EQ(v1[i], i);
}

TEST_F(pn_test, fill_transposed_range) {
  const unsigned col_count = 4;
  const unsigned log_row_count = 10;
  const unsigned cell_count = col_count << log_row_count;
  unsigned *v1;
  unsigned *v2;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v1, cell_count * sizeof(unsigned)));
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&v2, cell_count * sizeof(unsigned)));
  for (unsigned i = 0; i < cell_count; i++)
    v1[i] = i;
  ASSERT_CUDA_SUCCESS(pn::transpose<col_count>(v2, v1, log_row_count, nullptr));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  ASSERT_CUDA_SUCCESS(pn::fill_transposed_range(v1, col_count, log_row_count, nullptr));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  for (unsigned i = 0; i < cell_count; i++)
    ASSERT_EQ(v1[i], v2[i]);
}

TEST_F(pn_test, generate_permutation_polynomials_correctness) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  unsigned *indexes;
  const unsigned col_count = 4;
  const unsigned indexes_count = 3;
  const unsigned log_row_count = 10;
  const unsigned cell_count = col_count << log_row_count;
  const unsigned row_count = 1 << log_row_count;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&indexes, cell_count * sizeof(unsigned)));
  for (unsigned row = 0; row < row_count; row++)
    for (unsigned col = 0; col < col_count; col++) {
      indexes[col * row_count + row] = (row * col_count + col) % indexes_count;
    }
  pn::generate_permutation_polynomials_configuration cfg = {pool, stream, indexes, x, result1, col_count, log_row_count};
  ASSERT_CUDA_SUCCESS(pn::generate_permutation_polynomials(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  s *twiddles;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&twiddles, sizeof(s) << log_row_count));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(twiddles, log_row_count, 0, row_count, false, false, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned row = 0; row < row_count; row++)
    for (unsigned col = 0; col < col_count; col++) {
      const unsigned source_index = row * col_count + col;
      const unsigned target_index = source_index % indexes_count == 0 ? source_index
                                    : source_index < indexes_count    ? source_index + (cell_count - source_index - 1) / indexes_count * indexes_count
                                                                      : source_index - indexes_count;
      const unsigned target_row = target_index / col_count;
      const unsigned target_col = target_index % col_count;
      s expected = f::mul(twiddles[row], x[col]);
      s actual = result1[(target_col << log_row_count) + target_row];
      ASSERT_PRED2(f::eq, expected, actual);
    }
}

TEST_F(pn_test, generate_permutation_polynomials_benchmark) {
  const unsigned col_count = 4;
  const unsigned min_log_row_count = 19;
  const unsigned max_log_row_count = 26;
  const unsigned max_cell_count = col_count << max_log_row_count;
  unsigned *indexes;
  s *scalars;
  s *values;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&indexes, sizeof(unsigned) * max_cell_count));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&scalars, sizeof(s) * col_count));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&values, sizeof(s) * max_cell_count));
  fields_populate_random_device<f>(scalars, col_count);
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  preallocate_pool(pool, 25);
  printf("degree\t%11s\t%11s\n", "time", "memory");
  for (unsigned log_row_count = min_log_row_count; log_row_count <= max_log_row_count; log_row_count++) {
    size_t zero = 0;
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    const unsigned cell_count = col_count << log_row_count;
    populate_random_device(indexes, cell_count);
    trim_to_mask(indexes, (1 << (log_row_count - 1)) - 1, cell_count);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    cudaEvent_t start;
    cudaEvent_t end;
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    pn::generate_permutation_polynomials_configuration cfg = {pool, stream, indexes, scalars, values, col_count, log_row_count};
    ASSERT_CUDA_SUCCESS(pn::generate_permutation_polynomials(cfg));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    size_t used_mem;
    ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
    float elapsed;
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("2^%2d\t%8.3f ms\t%11zu\n", log_row_count, elapsed, used_mem);
    ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
    ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  }
}

TEST_F(pn_test, set_values_from_packed_bits) {
  const unsigned count = 123;
  const auto *bits = reinterpret_cast<const unsigned *>(x);
  ASSERT_CUDA_SUCCESS(pn::set_values_from_packed_bits(result1, bits, count, nullptr));
  for (unsigned i = 0; i < count; i++) {
    const bool bit = (bits[i / 32] >> (i % 32)) & 1;
    ASSERT_PRED2(f::eq, bit ? fd_q::get_one() : fd_q::storage{}, result1[i]);
  }
}
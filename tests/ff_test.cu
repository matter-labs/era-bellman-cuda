#include "bc.cuh"
#include "common.cuh"

#include "ff.cuh"
#include "ff_dispatch_st.cuh"
#include "ff_test_kernels.cuh"

class ff_test : public ::testing::Test {
protected:
  typedef fd_q f;
  typedef f::storage s;
  static const unsigned log_n = 16;
  static const unsigned n = 1 << log_n;

  ff_test() {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&x, n * sizeof(s)));
    assert(!cudaMallocManaged(&y, n * sizeof(s)));
    assert(!cudaMallocManaged(&zeroes, n * sizeof(s)));
    assert(!cudaMallocManaged(&ones, n * sizeof(s)));
    assert(!cudaMallocManaged(&twos, n * sizeof(s)));
    assert(!cudaMallocManaged(&result1, n * sizeof(s)));
    assert(!cudaMallocManaged(&result2, n * sizeof(s)));
  }

  ~ff_test() override {
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
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(x, n));
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(y, n));
    ASSERT_CUDA_SUCCESS(cudaMemset(zeroes, 0, n * sizeof(s)));
    ASSERT_CUDA_SUCCESS(fields_set<f>(ones, f::get_one(), n));
    ASSERT_CUDA_SUCCESS(fields_set<f>(twos, f::dbl(f::get_one()), n));
    ASSERT_CUDA_SUCCESS(cudaMemset(result1, 0, n * sizeof(f)));
    ASSERT_CUDA_SUCCESS(cudaMemset(result2, 0, n * sizeof(f)));
  }

  void TearDown() override { ASSERT_CUDA_SUCCESS(ff::tear_down()); }

  static unsigned bit_reverse(unsigned b) {
    unsigned mask = 0b11111111111111110000000000000000;
    b = (b & mask) >> 16 | (b & ~mask) << 16;
    mask = 0b11111111000000001111111100000000;
    b = (b & mask) >> 8 | (b & ~mask) << 8;
    mask = 0b11110000111100001111000011110000;
    b = (b & mask) >> 4 | (b & ~mask) << 4;
    mask = 0b11001100110011001100110011001100;
    b = (b & mask) >> 2 | (b & ~mask) << 2;
    mask = 0b10101010101010101010101010101010;
    b = (b & mask) >> 1 | (b & ~mask) << 1;
    return b;
  }

  s *x{};
  s *y{};
  s *zeroes{};
  s *ones{};
  s *twos{};
  s *result1{};
  s *result2{};

  void batch_inverse_correctness(const unsigned count, const bool in_place = false) {
    cudaMemPool_t pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    ASSERT_CUDA_SUCCESS(ff::naive_inverse<f>(x, result1, count, stream));
    auto output = in_place ? x : result2;
    ff::inverse_configuration cfg = {pool, stream, x, output, count};
    ASSERT_CUDA_SUCCESS(ff::inverse(cfg));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
    for (unsigned i = 0; i < n; i++)
      ASSERT_PRED2(f::eq, result1[i], output[i]);
  }
};

// Host x+0 == x
TEST_F(ff_test, HostXPlusZeroEqualsX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::add(x[i], zeroes[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Device x+0 == x
TEST_F(ff_test, DeviceXPlusZeroEqualsX) {
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, zeroes, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Host x-0 == x
TEST_F(ff_test, HostXMinusZeroEqualsX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::sub(x[i], zeroes[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Device x-0 == x
TEST_F(ff_test, DeviceXMinusZeroEqualsX) {
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, zeroes, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Host x-x == 0
TEST_F(ff_test, HostXMinusXEqualsZero) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::sub(x[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Device x-x == 0
TEST_F(ff_test, DeviceXMinusXEqualsZero) {
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, x, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Host -x+x == 0
TEST_F(ff_test, HostMinusXPlusXEqualsZero) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::neg(x[i]);
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::add(result1[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Device -x+x == 0
TEST_F(ff_test, DeviceMinusXPlusXEqualsZero) {
  ASSERT_CUDA_SUCCESS(fields_neg<f>(x, result1, n));
  ASSERT_CUDA_SUCCESS(fields_add<f>(result1, x, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Host x+y == y+x
TEST_F(ff_test, HostXPlusYEqualsYPlusX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::add(x[i], y[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::add(y[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x+y == y+x
TEST_F(ff_test, DeviceXPlusYEqualsYPlusX) {
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, y, result1, n));
  ASSERT_CUDA_SUCCESS(fields_add<f>(y, x, result2, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Host x-y == -(y-x)
TEST_F(ff_test, HostXMinusYEqualsMinusYMinusX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::sub(x[i], y[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::sub(y[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::neg(result2[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x-y == -(y-x)
TEST_F(ff_test, DeviceXMinusYEqualsMinusYMinusX) {
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, y, result1, n));
  ASSERT_CUDA_SUCCESS(fields_sub<f>(y, x, result2, n));
  ASSERT_CUDA_SUCCESS(fields_neg<f>(result2, result2, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x+y == Host x+y
TEST_F(ff_test, DeviceXPlusYEqualsHostXPlusY) {
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, y, result1, n));
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::add(x[i], y[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x-y == Host x-y
TEST_F(ff_test, DeviceXMinusYEqualsHostXMinusY) {
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, y, result1, n));
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::sub(x[i], y[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, AddPartialCarry) {
  x[0] = {1};
  y[0] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, y, result1, 1));
  result2[0] = {0, 0, 0, 0, 1, 0, 0, 0};
  ASSERT_PRED2(f::eq, result1[0], result2[0]);
}

TEST_F(ff_test, SubPartialCarry) {
  x[0] = {0, 0, 0, 0, 1, 0, 0, 0};
  y[0] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0};
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, y, result1, 1));
  result2[0] = {1};
  ASSERT_PRED2(f::eq, result1[0], result2[0]);
}

TEST_F(ff_test, AddFullCarry) {
  x[0] = {1};
  y[0] = f::get_modulus();
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, y, result1, 1));
  result2[0] = {1};
  ASSERT_PRED2(f::eq, result1[0], result2[0]);
}

TEST_F(ff_test, SubFullCarry) {
  x[0] = {1};
  y[0] = f::get_modulus();
  ASSERT_CUDA_SUCCESS(fields_sub<f>(x, y, result1, 1));
  result2[0] = {1};
  ASSERT_PRED2(f::eq, result1[0], result2[0]);
}

// Host x == from_montgomery(to_montgomery(x))
TEST_F(ff_test, HostXEqualsXToMontgomeryFromMontgomery) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::from_montgomery(f::to_montgomery(x[i]));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Device x == from_montgomery(to_montgomery(x))
TEST_F(ff_test, DeviceXEqualsXToMontgomeryFromMontgomery) {
  ASSERT_CUDA_SUCCESS(fields_to_montgomery<f>(x, result1, n));
  ASSERT_CUDA_SUCCESS(fields_from_montgomery<f>(result1, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Host x*0 == 0
TEST_F(ff_test, HostXTimesZeroEqualsZero) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::mul(x[i], zeroes[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Device x*0 == 0
TEST_F(ff_test, DeviceXTimesZeroEqualsZero) {
  ASSERT_CUDA_SUCCESS(fields_mul<f>(x, zeroes, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], zeroes[i]);
}

// Host x*1 == x
TEST_F(ff_test, HostXTimesOneEqualsX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::mul(x[i], ones[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Device x*1 == x
TEST_F(ff_test, DeviceXTimesOneEqualsX) {
  ASSERT_CUDA_SUCCESS(fields_mul<f>(x, ones, result1, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

// Host x*y == y*x
TEST_F(ff_test, HostXTimesYEqualsYTimesX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::mul(x[i], y[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::mul(y[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x*y == y*x
TEST_F(ff_test, DeviceXTimesYEqualsYTimesX) {
  ASSERT_CUDA_SUCCESS(fields_mul<f>(x, y, result1, n));
  ASSERT_CUDA_SUCCESS(fields_mul<f>(y, x, result2, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Host x*2 == x+x
TEST_F(ff_test, HostXTimesTwoEqualsXPlusX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::mul(x[i], twos[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::add(x[i], x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x*2 == x+x
TEST_F(ff_test, DeviceXTimesTwoEqualsXPlusX) {
  ASSERT_CUDA_SUCCESS(fields_mul<f>(x, twos, result1, n));
  ASSERT_CUDA_SUCCESS(fields_add<f>(x, x, result2, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Host x*2 == dbl(x)
TEST_F(ff_test, HostXTimesTwoEqualsDoubleX) {
  for (unsigned i = 0; i < n; i++)
    result1[i] = f::mul(x[i], twos[i]);
  for (unsigned i = 0; i < n; i++)
    result2[i] = f::dbl(x[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

// Device x*2 == dbl(x)
TEST_F(ff_test, DeviceXTimesTwoEqualsDoubleX) {
  ASSERT_CUDA_SUCCESS(fields_mul<f>(x, twos, result1, n));
  ASSERT_CUDA_SUCCESS(fields_dbl<f>(x, result2, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, onet_times_x_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  const auto one = f::get_one();
  ASSERT_CUDA_SUCCESS(ff::ax<f>(&one, x, result1, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

TEST_F(ff_test, two_times_x_equals_x_plus_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::ax<f>(twos, x, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(x, x, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, one_plus_x_equals_ones_plus_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::a_plus_x<f>(ones, x, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(ones, x, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, two_times_x_plus_y_equals_x_plus_x_plus_y_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  auto two = f::dbl(f::get_one());
  ASSERT_CUDA_SUCCESS(ff::ax_plus_y<f>(&two, x, y, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(x, x, result2, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(result2, y, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
}

TEST_F(ff_test, x_plus_zero_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(x, zeroes, result1, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

TEST_F(ff_test, x_plus_y_equals_y_plus_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(x, y, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(y, x, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, x_minus_zero_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_minus_y<f>(x, zeroes, result1, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], x[i]);
}

TEST_F(ff_test, x_minus_one_plus_one_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_minus_y<f>(x, ones, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(result1, ones, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result2[i], x[i]);
}

TEST_F(ff_test, x_minus_zero_times_y_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_minus_ay<f>(zeroes, x, y, result1, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], x[i]);
}

TEST_F(ff_test, x_minus_one_times_y_equals_x_minus_y_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_minus_ay<f>(ones, x, y, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_minus_y<f>(x, y, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, x_times_one_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(x, ones, result1, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, x[i], result1[i]);
}

TEST_F(ff_test, x_mul_y_equals_y_mul_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(x, y, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(y, x, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
}

TEST_F(ff_test, inverse_of_inverse_of_x_equals_x_host) {
  const unsigned count = min(n, 256);
  for (unsigned i = 0; i < count; i++)
    result1[i] = f::inverse(x[i]);
  for (unsigned i = 0; i < count; i++)
    result2[i] = f::inverse(result1[i]);
  for (unsigned i = 0; i < count; i++)
    ASSERT_PRED2(f::eq, result2[i], x[i]);
}

TEST_F(ff_test, inverse_of_inverse_of_x_equals_x_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::naive_inverse<f>(x, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::naive_inverse<f>(result1, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result2[i], x[i]);
}

TEST_F(ff_test, x_times_inverse_of_x_equals_one_host) {
  const unsigned count = min(n, 256);
  for (unsigned i = 0; i < count; i++)
    result1[i] = f::inverse(x[i]);
  for (unsigned i = 0; i < count; i++)
    result2[i] = f::mul(x[i], result1[i]);
  for (unsigned i = 0; i < count; i++)
    ASSERT_PRED2(f::eq, result2[i], ones[i]);
}

TEST_F(ff_test, x_times_inverse_of_x_equals_one_device) {
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::naive_inverse<f>(x, result1, n, stream));
  ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(x, result1, result2, n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result2[i], ones[i]);
}

TEST_F(ff_test, batch_inverse_equals_naive_inverse_size_n) { batch_inverse_correctness(n); }

TEST_F(ff_test, batch_inverse_equals_naive_inverse_size_odd) { batch_inverse_correctness(n - 33); }

TEST_F(ff_test, batch_inverse_equals_naive_inverse_size_less_than_full_wave) { batch_inverse_correctness(5555); }

TEST_F(ff_test, batch_inverse_equals_naive_inverse_in_place) { batch_inverse_correctness(n, true); }

TEST_F(ff_test, grand_product) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(s) * n, cudaMemcpyDefault));
  ff::grand_product_configuration cfg = {pool, stream, result1, result1, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  s product = f::CONFIG::one;
  for (unsigned i = 0; i < n; i++) {
    product = f::mul(product, x[i]);
    ASSERT_PRED2(f::eq, result1[i], product);
  }
}

TEST_F(ff_test, grand_product_reverse) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(s) * n, cudaMemcpyDefault));
  ff::grand_product_configuration cfg = {pool, stream, result1, result1, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product_reverse(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  s product = f::CONFIG::one;
  for (unsigned i = 0; i < n; i++) {
    unsigned index = n - i - 1;
    product = f::mul(product, x[index]);
    ASSERT_PRED2(f::eq, result1[index], product);
  }
}

TEST_F(ff_test, multiply_by_powers) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(s) * n, cudaMemcpyDefault));
  ff::multiply_by_powers_configuration cfg = {pool, stream, x, y, result1, n};
  ASSERT_CUDA_SUCCESS(ff::multiply_by_powers(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  s power = f::CONFIG::one;
  for (unsigned i = 0; i < n; i++) {
    s product = f::mul(x[i], power);
    ASSERT_PRED2(f::eq, result1[i], product);
    power = f::mul(power, y[0]);
  }
}

TEST_F(ff_test, poly_evaluate) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(s) * n, cudaMemcpyDefault));
  ff::poly_evaluate_configuration cfg = {pool, stream, x, y, result1, n};
  ASSERT_CUDA_SUCCESS(ff::poly_evaluate(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  s power = f::CONFIG::one;
  s sum = {};
  s point = y[0];
  for (unsigned i = 0; i < n; i++) {
    sum = f::add(sum, f::mul(x[i], power));
    power = f::mul(power, point);
  }
  ASSERT_PRED2(f::eq, result1[0], sum);
}

TEST_F(ff_test, get_powers_of_w) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(result1, 26, 0, n / 2, false, false, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(result1 + n / 2, 26, n / 2, n / 2, false, false, stream));
  ASSERT_CUDA_SUCCESS(ff::set_value_one<f>(result2, 1, stream));
  auto omega = f::sqr(f::sqr(f::CONFIG::omega));
  ASSERT_CUDA_SUCCESS(ff::set_value(result2 + 1, &omega, n - 1, stream));
  ff::grand_product_configuration cfg = {pool, stream, result2, result2, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, get_powers_of_w_inverse) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(result1, 26, 0, n / 2, true, false, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(result1 + n / 2, 26, n / 2, n / 2, true, false, stream));
  ASSERT_CUDA_SUCCESS(ff::set_value_one<f>(result2, 1, stream));
  auto omega = f::sqr(f::sqr(f::inverse(f::CONFIG::omega)));
  ASSERT_CUDA_SUCCESS(ff::set_value(result2 + 1, &omega, n - 1, stream));
  ff::grand_product_configuration cfg = {pool, stream, result2, result2, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, get_powers_of_g) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_g<f>(result1, log_n, 0, n / 2, false, false, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_g<f>(result1 + n / 2, log_n, n / 2, n / 2, false, false, stream));
  ASSERT_CUDA_SUCCESS(ff::set_value_one<f>(result2, 1, stream));
  auto g = f::to_montgomery({f::CONFIG::omega_generator});
  ASSERT_CUDA_SUCCESS(ff::set_value(result2 + 1, &g, n - 1, stream));
  ff::grand_product_configuration cfg = {pool, stream, result2, result2, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, get_powers_of_g_inverse) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_g<f>(result1, log_n, 0, n / 2, true, false, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_g<f>(result1 + n / 2, log_n, n / 2, n / 2, true, false, stream));
  ASSERT_CUDA_SUCCESS(ff::set_value_one<f>(result2, 1, stream));
  auto g = f::inverse(f::to_montgomery({f::CONFIG::omega_generator}));
  ASSERT_CUDA_SUCCESS(ff::set_value(result2 + 1, &g, n - 1, stream));
  ff::grand_product_configuration cfg = {pool, stream, result2, result2, n};
  ASSERT_CUDA_SUCCESS(ff::grand_product(cfg));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(f::eq, result1[i], result2[i]);
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, omega_shift) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result2, x, sizeof(f::storage) * n, cudaMemcpyDefault, stream));
  f::storage *omegas;
  ASSERT_CUDA_SUCCESS(cudaMallocFromPoolAsync(&omegas, sizeof(f::storage) * n, pool, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(omegas, 26, 0, n, false, false, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < 4; i++) {
    ASSERT_CUDA_SUCCESS(ff::omega_shift<f>(x, result1, 26, i, 0, n / 2, false, stream));
    ASSERT_CUDA_SUCCESS(ff::omega_shift<f>(x + n / 2, result1 + n / 2, 26, i, n / 2, n / 2, false, stream));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    for (unsigned j = 0; j < n; j++)
      ASSERT_PRED2(f::eq, result1[j], result2[j]);
    ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(result2, omegas, result2, n, stream));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, omega_shift_inverse) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result2, x, sizeof(f::storage) * n, cudaMemcpyDefault, stream));
  f::storage *omegas;
  ASSERT_CUDA_SUCCESS(cudaMallocFromPoolAsync(&omegas, sizeof(f::storage) * n, pool, stream));
  ASSERT_CUDA_SUCCESS(ff::get_powers_of_w<f>(omegas, 26, 0, n, true, false, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < 4; i++) {
    ASSERT_CUDA_SUCCESS(ff::omega_shift<f>(x, result1, 26, i, 0, n / 2, true, stream));
    ASSERT_CUDA_SUCCESS(ff::omega_shift<f>(x + n / 2, result1 + n / 2, 26, i, n / 2, n / 2, true, stream));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    for (unsigned j = 0; j < n; j++)
      ASSERT_PRED2(f::eq, result1[j], result2[j]);
    ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(result2, omegas, result2, n, stream));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, bit_reverse) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::bit_reverse<f::storage>(x, result1, log_n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++) {
    unsigned bi = bit_reverse(i) >> (32 - log_n);
    ASSERT_PRED2(f::eq, x[i], result1[bi]);
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, bit_reverse_in_place) {
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(f::storage) * n, cudaMemcpyDefault, stream));
  ASSERT_CUDA_SUCCESS(ff::bit_reverse<f::storage>(result1, result1, log_n, stream));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  for (unsigned i = 0; i < n; i++) {
    unsigned bi = bit_reverse(i) >> (32 - log_n);
    ASSERT_PRED2(f::eq, x[i], result1[bi]);
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
}

TEST_F(ff_test, bit_reverse_multigpu) {
  int deviceCount;
  ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2)
    GTEST_SKIP_("\nFound 1 visible device but the test requires 2.\n");
  cudaStream_t streams[2];
  ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&streams[0]));
  ASSERT_CUDA_SUCCESS(cudaSetDevice(1));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&streams[1]));
  const f::storage *values[] = {x, &x[n / 2]};
  f::storage *results[] = {result1, &result1[n / 2]};
  int device_ids[] = {0, 1};
  ASSERT_CUDA_SUCCESS(ff::bit_reverse_multigpu(values, results, log_n, streams, device_ids, 1));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(streams[0]));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(streams[1]));
  for (unsigned i = 0; i < n; i++) {
    unsigned bi = bit_reverse(i) >> (32 - log_n);
    ASSERT_PRED2(f::eq, x[i], result1[bi]);
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(streams[0]));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(streams[1]));
}

TEST_F(ff_test, bit_reverse_multigpu_in_place) {
  int deviceCount;
  ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2)
    GTEST_SKIP_("\nFound 1 visible device but the test requires 2.\n");
  cudaStream_t streams[2];
  ASSERT_CUDA_SUCCESS(cudaSetDevice(0));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&streams[0]));
  ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(result1, x, sizeof(f::storage) * n, cudaMemcpyDefault, streams[0]));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(streams[0]));
  ASSERT_CUDA_SUCCESS(cudaSetDevice(1));
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&streams[1]));
  const f::storage *values[] = {result1, &result1[n / 2]};
  f::storage *results[] = {result1, &result1[n / 2]};
  int device_ids[] = {0, 1};
  ASSERT_CUDA_SUCCESS(ff::bit_reverse_multigpu(values, results, log_n, streams, device_ids, 1));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(streams[0]));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(streams[1]));
  for (unsigned i = 0; i < n; i++) {
    unsigned bi = bit_reverse(i) >> (32 - log_n);
    ASSERT_PRED2(f::eq, x[i], result1[bi]);
  }
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(streams[0]));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(streams[1]));
}

TEST_F(ff_test, select) {
  unsigned *indexes = nullptr;
  ASSERT_CUDA_SUCCESS(cudaMallocManaged(&indexes, n * sizeof(unsigned)));
  for (unsigned i = 0; i < n; i++)
    indexes[i] = n - i - 1;
  ASSERT_CUDA_SUCCESS(ff::select(x, result1, indexes, n, nullptr));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  ASSERT_CUDA_SUCCESS(cudaFree(indexes));
  for (unsigned i = 0; i < n; i++) {
    ASSERT_PRED2(f::eq, x[i], result1[n - i - 1]);
  }
}

TEST_F(ff_test, ff_benchmark) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 26;
  const unsigned max_count = 1 << max_log_count;
  s a;
  fields_populate_random_host<f>(&a, 1);
  s *x;
  s *y;
  s *z;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&x, sizeof(s) * max_count));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&y, sizeof(s) * max_count));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&z, sizeof(s) * max_count));
  ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(x, max_count));
  ASSERT_CUDA_SUCCESS(fields_populate_random_device<f>(y, max_count));
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::inverse({pool, stream, x, z, max_count}));
  ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  cudaEvent_t start;
  cudaEvent_t end;
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
  printf("size\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\n", "ax", "x+y", "ax+y", "x-y", "ax-y", "x-ay", "x*y", "grand prod.", "poly eval.",
         "batch inv.");
  for (unsigned i = min_log_count; i <= max_log_count; i++) {
    printf("2^%2d", i);
    const unsigned count = 1 << i;
    float elapsed;
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::ax<f>(&a, x, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::x_plus_y<f>(x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::ax_plus_y<f>(&a, x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::x_minus_y<f>(x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::ax_minus_y<f>(&a, x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::x_minus_ay<f>(&a, x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::x_mul_y<f>(x, y, z, count, stream));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::grand_product({pool, stream, x, z, count}));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::poly_evaluate({pool, stream, x, &a, z, count}));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
    ASSERT_CUDA_SUCCESS(ff::inverse({pool, stream, x, z, count}));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
    ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
    ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
    printf("\t%8.3f ms", elapsed);
    printf("\n");
  }
  ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
  ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
  ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
  ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
  ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(pool));
  ASSERT_CUDA_SUCCESS(cudaFree(x));
  ASSERT_CUDA_SUCCESS(cudaFree(y));
}

TEST_F(ff_test, memory_requirements) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 28;
  const unsigned max_count = 1 << max_log_count;
  size_t zero = 0;
  s *values;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&values, sizeof(s) * max_count));
  cudaMemPool_t pool;
  ASSERT_CUDA_SUCCESS(bc::mem_pool_create(pool, 0));
  cudaStream_t stream;
  ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
  ASSERT_CUDA_SUCCESS(ff::inverse({pool, stream, values, values, max_count}));
  ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
  printf("size\t%11s\t%11s\t%11s\n", "grand prod.", "poly eval.", "batch inv.");
  for (unsigned i = min_log_count; i <= max_log_count; i++) {
    printf("2^%2d", i);
    const unsigned count = 1 << i;
    if (ff::grand_product({pool, stream, values, values, count}) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    s point = {};
    if (ff::poly_evaluate({pool, stream, values, &point, values, count}) == cudaSuccess) {
      size_t used_mem;
      ASSERT_CUDA_SUCCESS(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_mem));
      printf("\t%11zu", used_mem);
    } else
      printf("\t%11s", "N/A");
    ASSERT_CUDA_SUCCESS(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
    if (ff::inverse({pool, stream, values, values, count}) == cudaSuccess) {
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
}

#include "common.cuh"
#include "ec_test_kernels.cuh"

class ECTest : public ::testing::Test {
protected:
  static const unsigned n = 1 << 10;
  const pa g1 = pa::to_montgomery({{1}, {2}}, {});

  // x = 0x030644e7 2e131a02 9b85045b 68181585 d97816a9 16871ca8 d3c208c1 6d87cfd3
  // y = 0x15ed738c 0e0a7c92 e7845f96 b2ae9c0a 68a6a449 e3538fc7 ff3ebf7a 5a18a2c4
  const pa g2 = pa::to_montgomery({{0x6d87cfd3, 0xd3c208c1, 0x16871ca8, 0xd97816a9, 0x68181585, 0x9b85045b, 0x2e131a02, 0x030644e7},
                                   {0x5a18a2c4, 0xff3ebf7a, 0xe3538fc7, 0x68a6a449, 0xb2ae9c0a, 0xe7845f96, 0x0e0a7c92, 0x15ed738c}},
                                  {});

  // x = 0x0769bf9a c56bea3f f40232bc b1b6bd15 9315d847 15b8e679 f2d35596 1915abf0
  // y = 0x2ab799be e0489429 554fdb7c 8d086475 319e63b4 0b9c5b57 cdf1ff3d d9fe2261
  const pa g3 = pa::to_montgomery({{0x1915abf0, 0xf2d35596, 0x15b8e679, 0x9315d847, 0xb1b6bd15, 0xf40232bc, 0xc56bea3f, 0x0769bf9a},
                                   {0xd9fe2261, 0xcdf1ff3d, 0x0b9c5b57, 0x319e63b4, 0x8d086475, 0x554fdb7c, 0xe0489429, 0x2ab799be}},
                                  {});

  // x = 0x06a7b64a f8f414bc beef455b 1da5208c 9b592b83 ee659982 4caa6d2e e9141a76
  // y = 0x08e74e43 8cee31ac 104ce59b 94e45fe9 8a97d8f8 a6e75664 ce88ef5a 41e72fbc
  const pa g4 = pa::to_montgomery({{0xe9141a76, 0x4caa6d2e, 0xee659982, 0x9b592b83, 0x1da5208c, 0xbeef455b, 0xf8f414bc, 0x06a7b64a},
                                   {0x41e72fbc, 0xce88ef5a, 0xa6e75664, 0x8a97d8f8, 0x94e45fe9, 0x104ce59b, 0x8cee31ac, 0x08e74e43}},
                                  {});

  ECTest() {
    assert(!cudaDeviceReset());
    assert(!cudaMallocManaged(&g_p, n * sizeof(pp)));
    assert(!cudaMallocManaged(&g_z, n * sizeof(pz)));
    assert(!cudaMallocManaged(&zeroes_p, n * sizeof(pp)));
    assert(!cudaMallocManaged(&zeroes_z, n * sizeof(pz)));
    assert(!cudaMallocManaged(&ones_a, n * sizeof(pa)));
    assert(!cudaMallocManaged(&ones_p, n * sizeof(pp)));
    assert(!cudaMallocManaged(&ones_z, n * sizeof(pz)));
    assert(!cudaMallocManaged(&result1_p, n * sizeof(pp)));
    assert(!cudaMallocManaged(&result2_p, n * sizeof(pp)));
    assert(!cudaMallocManaged(&result1_z, n * sizeof(pz)));
    assert(!cudaMallocManaged(&result2_z, n * sizeof(pz)));
    assert(!cudaMallocManaged(&scalar, n * sizeof(fd_p::storage)));
  }

  ~ECTest() override {
    cudaFree(g_p);
    cudaFree(g_z);
    cudaFree(zeroes_p);
    cudaFree(zeroes_z);
    cudaFree(ones_a);
    cudaFree(ones_p);
    cudaFree(ones_z);
    cudaFree(result1_p);
    cudaFree(result2_p);
    cudaFree(result1_z);
    cudaFree(result2_z);
    cudaFree(scalar);
    cudaDeviceReset();
  }

  void SetUp() override {
    pp g1_p = pa::to_projective(g1, {});
    pp acc_p = pp::point_at_infinity({});
    pz g1_z = pa::to_xyzz(g1, {});
    pz acc_z = pz::point_at_infinity({});
    for (unsigned i = 0; i < n; i++) {
      g_p[i] = acc_p;
      g_z[i] = acc_z;
      zeroes_p[i] = pp::point_at_infinity({});
      zeroes_z[i] = pz::point_at_infinity({});
      ones_a[i] = g1;
      ones_p[i] = g1_p;
      ones_z[i] = g1_z;
      result1_p[i] = pp::point_at_infinity({});
      result2_p[i] = pp::point_at_infinity({});
      result1_z[i] = pz::point_at_infinity({});
      result2_z[i] = pz::point_at_infinity({});
      scalar[i] = {i};
      acc_p = curve::add(acc_p, g1, {});
      acc_z = curve::add(acc_z, g1, {});
    }
  }

  static bool pa_is_on_curve(const pa &point) { return pa::is_on_curve(point, {}); }

  static bool pp_is_on_curve(const pp &point) { return pp::is_on_curve(point, {}); }

  static bool pz_is_on_curve(const pz &point) { return pz::is_on_curve(point, {}); }

  static bool pp_eq(const pp &p1, const pp &p2) { return pp::eq(p1, p2, {}); }

  static bool pz_eq(const pz &p1, const pz &p2) { return pz::eq(p1, p2, {}); }

  pp *g_p{};
  pz *g_z{};
  pp *zeroes_p{};
  pz *zeroes_z{};
  pa *ones_a{};
  pp *ones_p{};
  pz *ones_z{};
  pp *result1_p{};
  pp *result2_p{};
  pz *result1_z{};
  pz *result2_z{};
  fd_q::storage *scalar{};
};

// constants are on curve
TEST_F(ECTest, GConstantsAreOnCurve) {
  ASSERT_PRED1(pa_is_on_curve, g1);
  ASSERT_PRED1(pa_is_on_curve, g2);
  ASSERT_PRED1(pa_is_on_curve, g3);
  ASSERT_PRED1(pa_is_on_curve, g4);
}

// g_p[0..4] equal precomputed values projective
TEST_F(ECTest, GsEqualPrecomputedValuesProjective) {
  ASSERT_PRED2(pp_eq, g_p[0], pp::point_at_infinity({}));
  ASSERT_PRED2(pp_eq, g_p[1], pa::to_projective(g1, {}));
  ASSERT_PRED2(pp_eq, g_p[2], pa::to_projective(g2, {}));
  ASSERT_PRED2(pp_eq, g_p[3], pa::to_projective(g3, {}));
  ASSERT_PRED2(pp_eq, g_p[4], pa::to_projective(g4, {}));
}

// Host x+0 == x projective
TEST_F(ECTest, HostXPlusZeroEqualsXProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::add(g_p[i], zeroes_p[i], {});
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i]);
}

// Device x+0 == x projective
TEST_F(ECTest, DeviceXPlusZeroEqualsXProjective) {
  ASSERT_CUDA_SUCCESS(ec_add(g_p, zeroes_p, result1_p, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i]);
}

// Host x[i]+1 affine == x[i+1] projective
TEST_F(ECTest, HostXIPlusOneAffineEqualsXIPlusOneProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::add(g_p[i], ones_a[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i - 1]);
}

// Device x[i]+1 affine == x[i+1] projective
TEST_F(ECTest, DeviceXIPlusOneAffineEqualsXIPlusOneProjective) {
  ASSERT_CUDA_SUCCESS(ec_add(g_p, ones_a, result1_p, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i - 1]);
}

// Host x[i]+1 projective == x[i+1] projective
TEST_F(ECTest, HostXIPlusOneProjectiveEqualsXIPlusOneProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::add(g_p[i], ones_p[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i - 1]);
}

// Device x[i]+1 projective == x[i+1] projective
TEST_F(ECTest, DeviceXIPlusOneProjectiveEqualsXIPlusOneProjective) {
  ASSERT_CUDA_SUCCESS(ec_add(g_p, ones_p, result1_p, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i], result1_p[i - 1]);
}

// Host x[i]-1 affine == x[i-1] projective
TEST_F(ECTest, HostXIMinusOneAffineEqualsXIMinusOneProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::sub(g_p[i], ones_a[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i - 1], result1_p[i]);
}

// Device x[i]-1 affine == x[i-1] projective
TEST_F(ECTest, DeviceXIMinusOneAffineEqualsXIMinusOneProjective) {
  ASSERT_CUDA_SUCCESS(ec_sub(g_p, ones_a, result1_p, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i - 1], result1_p[i]);
}

// Host x[i]-1 projective == x[i-1] projective
TEST_F(ECTest, HostXIMinusOneProjectiveEqualsXIMinusOneProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::sub(g_p[i], ones_p[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i - 1], result1_p[i]);
}

// Device x[i]-1 projective == x[i-1] projective
TEST_F(ECTest, DeviceXIMinusOneProjectiveEqualsXIMinusOneProjective) {
  ASSERT_CUDA_SUCCESS(ec_sub(g_p, ones_p, result1_p, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, g_p[i - 1], result1_p[i]);
}

// Host x+x=dbl(x) projective
TEST_F(ECTest, HostXPlusXEqualsDoubleXProjective) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = curve::add(g_p[i], g_p[i], {});
  for (unsigned i = 0; i < n; i++)
    result2_p[i] = curve::dbl(g_p[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result1_p[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result2_p[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], result2_p[i]);
}

// Device x+x=dbl(x) projective
TEST_F(ECTest, DeviceXPlusXEqualsDoubleXProjective) {
  ASSERT_CUDA_SUCCESS(ec_add(g_p, g_p, result1_p, n));
  ASSERT_CUDA_SUCCESS(ec_dbl(g_p, result2_p, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result1_p[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result2_p[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], result2_p[i]);
}

// Host g1*i == g[i] projective
TEST_F(ECTest, HostG1TimesIEqualsGIProjective) {
  const pp &g1_p = pa::to_projective(g1, {});
  for (unsigned i = 0; i < n; i++) {
    result1_p[i] = curve::mul<fd_q>(scalar[i], g1_p, {});
  }
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], g_p[i]);
}

// Device g1*i == g[i] projective
TEST_F(ECTest, DeviceG1TimesIEqualsGIProjective) {
  ASSERT_CUDA_SUCCESS(ec_mul(scalar, ones_p, result1_p, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], g_p[i]);
}

// g_z[0..4] equal precomputed values xyzz
TEST_F(ECTest, GsEqualPrecomputedValuesXyzz) {
  ASSERT_PRED2(pz_eq, g_z[0], pz::point_at_infinity({}));
  ASSERT_PRED2(pz_eq, g_z[1], pa::to_xyzz(g1, {}));
  ASSERT_PRED2(pz_eq, g_z[2], pa::to_xyzz(g2, {}));
  ASSERT_PRED2(pz_eq, g_z[3], pa::to_xyzz(g3, {}));
  ASSERT_PRED2(pz_eq, g_z[4], pa::to_xyzz(g4, {}));
}

// Host x+0 == x xyzz
TEST_F(ECTest, HostXPlusZeroEqualsXXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::add(g_z[i], zeroes_z[i], {});
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i]);
}

// Device x+0 == x xyzz
TEST_F(ECTest, DeviceXPlusZeroEqualsXXyzz) {
  ASSERT_CUDA_SUCCESS(ec_add(g_z, zeroes_z, result1_z, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i]);
}

// Host x[i]+1 affine == x[i+1] xyzz
TEST_F(ECTest, HostXIPlusOneAffineEqualsXIPlusOneXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::add(g_z[i], ones_a[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i - 1]);
}

// Device x[i]+1 affine == x[i+1] xyzz
TEST_F(ECTest, DeviceXIPlusOneAffineEqualsXIPlusOneXyzz) {
  ASSERT_CUDA_SUCCESS(ec_add(g_z, ones_a, result1_z, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i - 1]);
}

// Host x[i]+1 xyzz == x[i+1] xyzz
TEST_F(ECTest, HostXIPlusOneXyzzEqualsXIPlusOneXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::add(g_z[i], ones_z[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i - 1]);
}

// Device x[i]+1 xyzz == x[i+1] xyzz
TEST_F(ECTest, DeviceXIPlusOneXyzzEqualsXIPlusOneXyzz) {
  ASSERT_CUDA_SUCCESS(ec_add(g_z, ones_z, result1_z, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i], result1_z[i - 1]);
}

// Host x[i]-1 affine == x[i-1] xyzz
TEST_F(ECTest, HostXIMinusOneAffineEqualsXIMinusOneXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::sub(g_z[i], ones_a[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i - 1], result1_z[i]);
}

// Device x[i]-1 affine == x[i-1] xyzz
TEST_F(ECTest, DeviceXIMinusOneAffineEqualsXIMinusOneXyzz) {
  ASSERT_CUDA_SUCCESS(ec_sub(g_z, ones_a, result1_z, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i - 1], result1_z[i]);
}

// Host x[i]-1 xyzz == x[i-1] xyzz
TEST_F(ECTest, HostXIMinusOneXyzzEqualsXIMinusOneXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::sub(g_z[i], ones_z[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i - 1], result1_z[i]);
}

// Device x[i]-1 xyzz == x[i-1] xyzz
TEST_F(ECTest, DeviceXIMinusOneXyzzEqualsXIMinusOneXyzz) {
  ASSERT_CUDA_SUCCESS(ec_sub(g_z, ones_z, result1_z, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, g_z[i - 1], result1_z[i]);
}

// Host x+x=dbl(x) xyzz
TEST_F(ECTest, HostXPlusXEqualsDoubleXXyzz) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = curve::add(g_z[i], g_z[i], {});
  for (unsigned i = 0; i < n; i++)
    result2_z[i] = curve::dbl(g_z[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result1_z[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result2_z[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], result2_z[i]);
}

// Device x+x=dbl(x) xyzz
TEST_F(ECTest, DeviceXPlusXEqualsDoubleXXyzz) {
  ASSERT_CUDA_SUCCESS(ec_add(g_z, g_z, result1_z, n));
  ASSERT_CUDA_SUCCESS(ec_dbl(g_z, result2_z, n));
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result1_z[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result2_z[i]);
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], result2_z[i]);
}

// Host g1*i == g[i] xyzz
TEST_F(ECTest, HostG1TimesIEqualsGIXyzz) {
  const pz &g1_z = pa::to_xyzz(g1, {});
  for (unsigned i = 0; i < n; i++) {
    result1_z[i] = curve::mul<fd_q>(scalar[i], g1_z, {});
  }
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], g_z[i]);
}

// Device g1*i == g[i] xyzz
TEST_F(ECTest, DeviceG1TimesIEqualsGIXyzz) {
  ASSERT_CUDA_SUCCESS(ec_mul(scalar, ones_z, result1_z, n));
  for (unsigned i = 0; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], g_z[i]);
}

TEST_F(ECTest, XProjectiveToXyzzToProjectiveEqualsX) {
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = pp::to_xyzz(g_p[i], {});
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = pz::to_projective(result1_z[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result1_z[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result1_p[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], g_z[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], g_p[i]);
}

TEST_F(ECTest, XXyzzToProjectiveToXyzzEqualsX) {
  for (unsigned i = 0; i < n; i++)
    result1_p[i] = pz::to_projective(g_z[i], {});
  for (unsigned i = 0; i < n; i++)
    result1_z[i] = pp::to_xyzz(result1_p[i], {});
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pp_is_on_curve, result1_p[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED1(pz_is_on_curve, result1_z[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pp_eq, result1_p[i], g_p[i]);
  for (unsigned i = 1; i < n; i++)
    ASSERT_PRED2(pz_eq, result1_z[i], g_z[i]);
}
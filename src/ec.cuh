#pragma once

#include "common.cuh"
#include "ff_config.cuh"
#include "memory.cuh"
#include "ptx.cuh"
#include <cstdint>

template <typename FD, unsigned B_VALUE> struct ec {
  typedef FD field;
  typedef typename FD::storage storage;
  typedef typename FD::storage_wide storage_wide;

  struct point_projective;

  struct point_jacobian;

  struct point_xyzz;

  struct point_affine {
    storage x;
    storage y;

    static __host__ __device__ __forceinline__ point_affine to_montgomery(const point_affine &point, const FD &fd) {
      const storage x = fd.to_montgomery(point.x);
      const storage y = fd.to_montgomery(point.y);
      return {x, y};
    }

    static __host__ __device__ __forceinline__ point_affine from_montgomery(const point_affine &point, const FD &fd) {
      const storage x = fd.from_montgomery(point.x);
      const storage y = fd.from_montgomery(point.y);
      return {x, y};
    }

    static __host__ __device__ __forceinline__ point_affine neg(const point_affine &point, const FD &fd) { return {point.x, fd.neg(point.y)}; }

    static __host__ __device__ __forceinline__ point_projective to_projective(const point_affine &point, const FD &fd) {
      return {point.x, point.y, fd.get_one()};
    }

    static __host__ __device__ __forceinline__ point_jacobian to_jacobian(const point_affine &point, const FD &fd) { return {point.x, point.y, fd.get_one()}; }

    static __host__ __device__ __forceinline__ point_xyzz to_xyzz(const point_affine &point, const FD &fd) {
      return {point.x, point.y, fd.get_one(), fd.get_one()};
    }

    //  y^2=x^3+b
    static __host__ __device__ __forceinline__ bool is_on_curve(const point_affine &point, const FD &fd) {
      const storage x = point.x;
      const storage y = point.y;
      const storage y2 = fd.mul(y, y);
      const storage x3 = fd.mul(x, fd.template sqr<0>(x));
      const storage a = y2;
      const storage b = fd.add(x3, fd.mul(B_VALUE, fd.get_one()));
      return fd.eq(a, b);
    }
  };

  struct point_projective {
    storage x;
    storage y;
    storage z;

    static constexpr __host__ __device__ __forceinline__ point_projective point_at_infinity(const FD &fd) { return {{0}, fd.get_one(), {0}}; };

    static __host__ __device__ __forceinline__ point_projective to_montgomery(const point_projective &point, const FD &fd) {
      const storage x = fd.to_montgomery<0>(point.x);
      const storage y = fd.to_montgomery<0>(point.y);
      const storage z = fd.to_montgomery<0>(point.z);
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_projective from_montgomery(const point_projective &point, const FD &fd) {
      const storage x = fd.from_montgomery<0>(point.x);
      const storage y = fd.from_montgomery<0>(point.y);
      const storage z = fd.from_montgomery<0>(point.z);
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_projective neg(const point_projective &point, const FD &fd) {
      return {point.x, fd.template neg<2>(point.y), point.z};
    }

    static __host__ __device__ __forceinline__ bool eq(const point_projective &p1, const point_projective &p2, const FD &fd) {
      const storage z1 = fd.reduce(p1.z);
      const storage z2 = fd.reduce(p2.z);
      if (fd.is_zero(z1) != fd.is_zero(z2))
        return false;
      const storage x1 = fd.mul(p1.x, z2);
      const storage x2 = fd.mul(p2.x, z1);
      const storage y1 = fd.mul(p1.y, z2);
      const storage y2 = fd.mul(p2.y, z1);
      const bool eqx = fd.eq(x1, x2);
      const bool eqy = fd.eq(y1, y2);
      return eqx && eqy;
    }

    static __host__ __device__ __forceinline__ point_jacobian to_jacobian(const point_projective &point, const FD &fd) {
      const storage x = fd.template mul<0>(point.x, point.z);
      const storage y = fd.template mul<0>(point.y, fd.template sqr<0>(point.z));
      const storage z = point.z;
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_xyzz to_xyzz(const point_projective &point, const FD &fd) {
      const storage z = point.z;
      const storage zz = fd.template sqr<0>(z);
      const storage x = fd.template mul<0>(point.x, z);
      const storage y = fd.template mul<0>(point.y, zz);
      const storage zzz = fd.template mul<0>(z, zz);

      return {x, y, zz, zzz};
    }

    // x=X/Z
    // y=Y/Z
    // y^2=x^3+b => Y^2/Z^2=X^3/Z^3+b => Y^2*Z = X^3 + b*Z^3
    static __host__ __device__ __forceinline__ bool is_on_curve(const point_projective &point, const FD &fd) {
      const storage x = point.x;
      const storage y = point.y;
      const storage z = fd.reduce(point.z);
      if (fd.is_zero(z))
        return false;
      const storage y2 = fd.template mul<0>(y, y);
      const storage x3 = fd.mul(x, fd.template sqr<0>(x));
      const storage z3 = fd.mul(z, fd.template sqr<0>(z));
      const storage a = fd.mul(y2, z);
      const storage b = fd.add(x3, fd.mul(B_VALUE, z3));
      return fd.eq(a, b);
    }
  };

  struct point_jacobian {
    storage x;
    storage y;
    storage z;

    static constexpr __host__ __device__ __forceinline__ point_jacobian point_at_infinity(const FD &fd) { return {{0}, fd.get_one(), {0}}; };

    static __host__ __device__ __forceinline__ point_jacobian to_montgomery(const point_jacobian &point, const FD &fd) {
      const storage x = fd.to_montgomery<0>(point.x);
      const storage y = fd.to_montgomery<0>(point.y);
      const storage z = fd.to_montgomery<0>(point.z);
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_jacobian from_montgomery(const point_jacobian &point, const FD &fd) {
      const storage x = fd.from_montgomery<0>(point.x);
      const storage y = fd.from_montgomery<0>(point.y);
      const storage z = fd.from_montgomery<0>(point.z);
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_jacobian neg(const point_jacobian &point, const FD &fd) {
      return {point.x, fd.template neg<2>(point.y), point.z};
    }

    static __host__ __device__ __forceinline__ bool eq(const point_jacobian &p1, const point_jacobian &p2, const FD &fd) {
      const storage z1 = fd.reduce(p1.z);
      const storage z2 = fd.reduce(p2.z);
      if (fd.is_zero(z1) != fd.is_zero(z2))
        return false;
      const storage z1z1 = fd.template sqr<0>(z1);
      const storage z2z2 = fd.template sqr<0>(z2);
      const storage z1z1z1 = fd.template mul<0>(z1, z1z1);
      const storage z2z2z2 = fd.template mul<0>(z2, z2z2);
      const storage x1 = fd.mul(p1.x, z2z2);
      const storage x2 = fd.mul(p2.x, z1z1);
      const storage y1 = fd.mul(p1.y, z2z2z2);
      const storage y2 = fd.mul(p2.y, z1z1z1);
      return fd.eq(x1, x2) && fd.eq(y1, y2);
    }

    static __host__ __device__ __forceinline__ point_projective to_projective(const point_jacobian &point, const FD &fd) {
      const storage x = fd.template mul<0>(point.x, point.z);
      const storage y = point.y;
      const storage z = fd.template mul<0>(point.z, fd.template sqr<0>(point.z));
      return {x, y, z};
    }

    // x=X/Z^2
    // y=Y/Z^3
    // y^2=x^3+b => Y^2/Z^6=X^3/Z^6+b => Y^2 = X^3 + b*Z^6
    static bool __host__ __device__ __forceinline__ is_on_curve(const point_jacobian &point, const FD &fd) {
      const storage x = point.x;
      const storage y = point.y;
      const storage z = fd.reduce(point.z);
      if (fd.is_zero(z))
        return false;
      const storage y2 = fd.mul(y, y);
      const storage x3 = fd.mul(x, fd.template sqr<0>(x));
      const storage z2 = fd.sqr(z);
      const storage z6 = fd.mul(z2, fd.template sqr<0>(z2));
      const storage a = y2;
      const storage b = fd.add(x3, fd.mul(B_VALUE, z6));
      return fd.eq(a, b);
    }
  };

  //  https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html
  //  x=X/ZZ
  //  y=Y/ZZZ
  //  ZZ^3=ZZZ^2
  struct point_xyzz {
    storage x;
    storage y;
    storage zz;
    storage zzz;

    static constexpr __host__ __device__ __forceinline__ point_xyzz point_at_infinity(const FD &fd) { return {{0}, fd.get_one(), {0}, {0}}; };

    static __host__ __device__ __forceinline__ point_xyzz to_montgomery(const point_xyzz &point, const FD &fd) {
      const storage x = fd.to_montgomery<0>(point.x);
      const storage y = fd.to_montgomery<0>(point.y);
      const storage zz = fd.to_montgomery<0>(point.zz);
      const storage zzz = fd.to_montgomery<0>(point.zzz);
      return {x, y, zz, zzz};
    }

    static __host__ __device__ __forceinline__ point_xyzz from_montgomery(const point_xyzz &point, const FD &fd) {
      const storage x = fd.from_montgomery<0>(point.x);
      const storage y = fd.from_montgomery<0>(point.y);
      const storage zz = fd.from_montgomery<0>(point.zz);
      const storage zzz = fd.from_montgomery<0>(point.zzz);
      return {x, y, zz, zzz};
    }

    static __host__ __device__ __forceinline__ point_xyzz neg(const point_xyzz &point, const FD &fd) {
      return {point.x, fd.template neg<2>(point.y), point.zz, point.zzz};
    }

    static __host__ __device__ __forceinline__ bool eq(const point_xyzz &p1, const point_xyzz &p2, const FD &fd) {
      const storage zz1 = fd.reduce(p1.zz);
      const storage zz2 = fd.reduce(p2.zz);
      if (fd.is_zero(zz1) != fd.is_zero(zz2))
        return false;
      const storage x1 = fd.mul(p1.x, p2.zz);
      const storage x2 = fd.mul(p2.x, p1.zz);
      const storage y1 = fd.mul(p1.y, p2.zzz);
      const storage y2 = fd.mul(p2.y, p1.zzz);
      return fd.eq(x1, x2) && fd.eq(y1, y2);
    }

    static __host__ __device__ __forceinline__ point_projective to_projective(const point_xyzz &point, const FD &fd) {
      const storage z2 = fd.reduce(point.zz);
      if (fd.is_zero(z2))
        return point_projective::point_at_infinity(fd);
      const storage x = fd.template mul<0>(point.x, point.zzz);
      const storage y = fd.template mul<0>(point.y, point.zz);
      const storage z = fd.template mul<0>(point.zz, point.zzz);
      return {x, y, z};
    }

    static __host__ __device__ __forceinline__ point_jacobian to_jacobian(const point_xyzz &point, const FD &fd) {
      const storage zz = fd.reduce(point.zz);
      if (fd.is_zero(zz))
        return point_jacobian::point_at_infinity(fd);
      const storage z = fd.template mul<0>(point.zz, point.zzz);
      const storage x = fd.template mul<0>(fd.template mul<0>(point.x, point.zzz), z);
      const storage y = fd.template mul<0>(fd.template mul<0>(point.y, point.zz), fd.template sqr<0>(z));
      return {x, y, z};
    }

    // x=X/Z^2
    // y=Y/Z^3
    // y^2=x^3+b => Y^2/Z^6=X^3/Z^6+b => Y^2 = X^3 + b*Z^6
    static bool __host__ __device__ __forceinline__ is_on_curve(const point_xyzz &point, const FD &fd) {
      const storage x = point.x;
      const storage y = point.y;
      const storage z3 = fd.reduce(point.zzz);
      if (fd.is_zero(z3))
        return false;
      const storage y2 = fd.mul(y, y);
      const storage x3 = fd.mul(x, fd.template sqr<0>(x));
      const storage z6 = fd.sqr(z3);
      const storage a = y2;
      const storage b = fd.add(x3, fd.mul(B_VALUE, z6));
      return fd.eq(a, b);
    }
  };

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
  template <bool CHECK_ZERO = true> static __host__ __device__ __forceinline__ point_jacobian dbl_2009_l(const point_jacobian &point, const FD &fd) {
    const storage X1 = point.x;
    const storage Y1 = point.y;
    const storage Z1 = point.z;
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(Z1))))
        return point;
    }
    const storage A = fd.template sqr<0>(X1);                                         // A = X1^2
    const storage B = fd.template sqr<0>(Y1);                                         // B = Y1^2
    const storage C = fd.template sqr<0>(B);                                          // C = B^2
    const storage t0 = fd.template add<2>(X1, B);                                     // t0 = X1+B
    const storage t1 = fd.template sqr<0>(t0);                                        // t1 = t0^2
    const storage t2 = fd.template sub<2>(t1, A);                                     // t2 = t1-A
    const storage t3 = fd.template sub<2>(t2, C);                                     // t3 = t2-C
    const storage D = fd.template dbl<2>(t3);                                         // D = 2*t3
    const storage E = fd.template add<2>(A, fd.template dbl<2>(A));                   // E = 3*A
    const storage F = fd.template sqr<0>(E);                                          // F = E^2
    const storage t4 = fd.template dbl<2>(D);                                         // t4 = 2*D
    const storage X3 = fd.template sub<2>(F, t4);                                     // X3 = F-t4
    const storage t5 = fd.template sub<2>(D, X3);                                     // t5 = D-X3
    const storage t6 = fd.template dbl<2>(fd.template dbl<2>(fd.template dbl<2>(C))); // t6 = 8*C
    const storage t7 = fd.template mul<0>(t5, E);                                     // t7 = E*t5
    const storage Y3 = fd.template sub<2>(t7, t6);                                    // Y3 = t7-t6
    const storage t8 = fd.template mul<0>(Z1, Y1);                                    // t8 = Y1*Z1
    const storage Z3 = fd.template dbl<2>(t8);                                        // Z3 = 2*t8
    return {X3, Y3, Z3};
  }

  template <bool CHECK_ZERO = true> static __host__ __device__ __forceinline__ point_jacobian dbl(const point_jacobian &point, const FD &fd) {
    return dbl_2009_l<CHECK_ZERO>(point, fd);
  }

  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian add_2007_bl(const point_jacobian &p1, const point_jacobian &p2, const FD &fd) {
    const storage X1 = p1.x;
    const storage Y1 = p1.y;
    const storage Z1 = p1.z;
    const storage X2 = p2.x;
    const storage Y2 = p2.y;
    const storage Z2 = p2.z;
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(Z1))))
        return p2;
      if (unlikely(fd.is_zero(fd.reduce(Z2))))
        return p1;
    }
    const storage Z1Z1 = fd.template sqr<0>(Z1);     // Z1Z1 = Z1^2
    const storage Z2Z2 = fd.template sqr<0>(Z2);     // Z2Z2 = Z2^2
    const storage U1 = fd.template mul<0>(Z2Z2, X1); // U1 = X1*Z2Z2
    const storage U2 = fd.template mul<0>(Z1Z1, X2); // U2 = X2*Z1Z1
    const storage t0 = fd.template mul<0>(Z2Z2, Z2); // t0 = Z2*Z2Z2
    const storage S1 = fd.template mul<0>(t0, Y1);   // S1 = Y1*t0
    const storage t1 = fd.template mul<0>(Z1Z1, Z1); // t1 = Z1*Z1Z1
    const storage S2 = fd.template mul<0>(t1, Y2);   // S2 = Y2*t1
    const storage H = fd.template sub<2>(U2, U1);    // H = U2-U1
    const storage t3 = fd.template sub<2>(S2, S1);   // t3 = S2-S1
    if (CHECK_DOUBLE) {
      if (unlikely(fd.is_zero(fd.reduce(H))) && unlikely(fd.is_zero(fd.reduce(t3))))
        return dbl<false>(p1, fd);
    }
    const storage t2 = fd.template dbl<2>(H);          // t2 = 2*H
    const storage I = fd.template sqr<0>(t2);          // I = t2^2
    const storage J = fd.template mul<0>(I, H);        // J = H*I
    const storage R = fd.template dbl<2>(t3);          // R = 2*t3
    const storage V = fd.template mul<0>(I, U1);       // V = U1*I
    const storage t4 = fd.template sqr<0>(R);          // t4 = R^2
    const storage t5 = fd.template dbl<2>(V);          // t5 = 2*V
    const storage t6 = fd.template sub<2>(t4, J);      // t6 = t4-J
    const storage X3 = fd.template sub<2>(t6, t5);     // X3 = t6-t5
    const storage t7 = fd.template sub<2>(V, X3);      // t7 = V-X3
    const storage t8 = fd.template mul<0>(J, S1);      // t8 = S1*J
    const storage t9 = fd.template dbl<2>(t8);         // t9 = 2*t8
    const storage t10 = fd.template mul<0>(t7, R);     // t10 = R*t7
    const storage Y3 = fd.template sub<2>(t10, t9);    // Y3 = t10-t9
    const storage t11 = fd.template add<2>(Z1, Z2);    // t11 = Z1+Z2
    const storage t12 = fd.template sqr<0>(t11);       // t12 = t11^2
    const storage t13 = fd.template sub<2>(t12, Z1Z1); // t13 = t12-Z1Z1
    const storage t14 = fd.template sub<2>(t13, Z2Z2); // t14 = t13-Z2Z2
    const storage Z3 = fd.template mul<0>(H, t14);     // Z3 = t14*H
    return {X3, Y3, Z3};
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian add(const point_jacobian &p1, const point_jacobian &p2, const FD &fd) {
    return add_2007_bl<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
  }

  // https://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2008-g
  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian add_madd_2008_q(const point_jacobian &p1, const point_affine &p2, const FD &fd) {
    const storage X1 = p1.x; // < 2
    const storage Y1 = p1.y; // < 2
    const storage Z1 = p1.z; // < 2
    const storage X2 = p2.x; // < 1
    const storage Y2 = p2.y; // < 1
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(Z1))))
        return point_affine::to_jacobian(p2, fd);
    }
    storage T1 = fd.template sqr<0>(Z1);     // T1 = Z1^2    < 2
    storage T2 = fd.template mul<0>(T1, Z1); // T2 = T1*Z1   < 2
    T1 = fd.template mul<0>(T1, X2);         // T1 = T1*X2   < 2
    T2 = fd.template mul<0>(T2, Y2);         // T2 = T2*Y2   < 2
    T1 = fd.template sub<2>(X1, T1);         // T1 = X1-T1   < 2
    T2 = fd.template sub<2>(T2, Y1);         // T2 = T2-Y1   < 2
    if (CHECK_DOUBLE) {
      if (unlikely(fd.is_zero(fd.reduce(T1))) && unlikely(fd.is_zero(fd.reduce(T2))))
        return dbl<false>(p1, fd);
    }
    storage Z3 = fd.template mul<0>(Z1, T1);                // Z3 = Z1*T1   < 2
    storage T4 = fd.template sqr<0>(T1);                    // T4 = T1^2    < 2
    T1 = fd.template mul<0>(T1, T4);                        // T1 = T1*T4   < 2
    T4 = fd.template mul<0>(T4, X1);                        // T4 = T4*X1   < 2
    storage X3 = fd.template sqr<0>(T2);                    // X3 = T2^2    < 2
    X3 = fd.template add<0>(X3, T1);                        // X3 = X3+T1   < 4
    storage Y3 = fd.template mul<0>(T1, Y1);                // Y3 = T1*Y1   < 2
    T1 = fd.template dbl<0>(T4);                            // T1 = 2*T4    < 4
    X3 = fd.template reduce<2>(fd.template sub<4>(X3, T1)); // X3 = X3-T1   < 2
    T4 = fd.template sub<2>(X3, T4);                        // T4 = X3-T4   < 2
    T4 = fd.template mul<0>(T4, T2);                        // T4 = T4*T2   < 2
    Y3 = fd.template sub<2>(T4, Y3);                        // Y3 = T4-Y3   < 2
    return {X3, Y3, Z3};
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian add(const point_jacobian &p1, const point_affine &p2, const FD &fd) {
    return add_madd_2008_q<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-mdbl-2008-s-1
  static __host__ __device__ __forceinline__ point_xyzz mdbl_2008_s_1(const point_affine &point, const FD &fd) {
    const storage Y1 = point.y;                                        //  < 2
    const storage X1 = point.x;                                        //  < 2
    const storage U = fd.template dbl<2>(Y1);                          // U = 2*Y1      < 2
    const storage V = fd.template sqr<0>(U);                           // V = U^2       < 2
    const storage W = fd.template mul<0>(U, V);                        // W = U*V       < 2
    const storage S = fd.template mul<0>(X1, V);                       // S = X1*V      < 2
    const storage t0 = fd.template sqr<1>(X1);                         // t0 = X1^2     < 1
    const storage t1 = fd.template add<0>(t0, fd.template dbl<1>(t0)); // t1 = 3*t0     < 2
    const storage M = t1;                                              // M = t1+a      < 2
    const storage t2 = fd.template sqr<0>(M);                          // t2 = M^2      < 2
    const storage t3 = fd.template dbl<2>(S);                          // t3 = 2*S      < 2
    const storage X3 = fd.template sub<2>(t2, t3);                     // X3 = t2-t3    < 2
    const storage t4 = fd.template sub<2>(S, X3);                      // t4 = S-X3     < 2
#ifdef __CUDA_ARCH__
    // Y3 = M*t4 - W*Y1
    const storage_wide t5_wide = fd.template mul_wide<0>(W, Y1);   // < 4*mod^2 (Y1 may be in [0, 2*mod))
    const storage_wide t6_wide = fd.template mul_wide<0>(M, t4);   // < 4*mod^2
    storage_wide diff = fd.template sub_wide<4>(t6_wide, t5_wide); // < 4*mod^2
    fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
    const storage Y3 = diff.get_lo();                              // < 2*mod
#else
    const storage t5 = fd.template mul<0>(W, Y1);     // t5 = W*Y1     < 2
    const storage t6 = fd.template mul<0>(M, t4);     // t6 = M*t4     < 2
    const storage Y3 = fd.template sub<2>(t6, t5);    // Y3 = t6-t5    < 2
#endif
    const storage ZZ3 = V;  // ZZ3 = V       < 2
    const storage ZZZ3 = W; // ZZZ3 = W      < 2
    return {X3, Y3, ZZ3, ZZZ3};
  }

  static __host__ __device__ __forceinline__ point_xyzz dbl(const point_affine &point, const FD &fd) { return mdbl_2008_s_1(point, fd); }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#doubling-dbl-2008-s-1
  template <bool CHECK_ZERO = true> static __host__ __device__ __forceinline__ point_xyzz dbl_2008_s_1(const point_xyzz &point, const FD &fd) {
    const storage X1 = point.x;     // < 2
    const storage Y1 = point.y;     // < 2
    const storage ZZ1 = point.zz;   // < 2
    const storage ZZZ1 = point.zzz; // < 2
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        return point;
    }
    const storage U = fd.template dbl<2>(Y1);                          // U = 2*Y1       < 2
    const storage V = fd.template sqr<0>(U);                           // V = U^2        < 2
    const storage W = fd.template mul<0>(U, V);                        // W = U*V        < 2
    const storage S = fd.template mul<0>(X1, V);                       // S = X1*V       < 2
    const storage t0 = fd.template sqr<1>(X1);                         // t0 = X1^2      < 1
                                                                       // t1 = ZZ1^2          unused
                                                                       // t2 = a*t1           a=0 => t2 = 0
    const storage t3 = fd.template add<0>(t0, fd.template dbl<1>(t0)); // t3 = 3*t0      < 2
    const storage M = t3;                                              // M = t3+t2      < 2  t2 = 0 => M = t3
    const storage t4 = fd.template sqr<0>(M);                          // t4 = M^2       < 2
    const storage t5 = fd.template dbl<2>(S);                          // t5 = 2*S       < 2
    const storage X3 = fd.template sub<2>(t4, t5);                     // X3 = t4-t5     < 2
    const storage t6 = fd.template sub<2>(S, X3);                      // t6 = S-X3      < 2
#ifdef __CUDA_ARCH__
    // Y3 = M*t6 - W*Y1
    const storage_wide t7_wide = fd.template mul_wide<0>(W, Y1);   // < 4*mod^2 (Y1 may be in [0, 2*mod))
    const storage_wide t8_wide = fd.template mul_wide<0>(M, t6);   // < 4*mod^2
    storage_wide diff = fd.template sub_wide<4>(t8_wide, t7_wide); // < 4*mod^2
    fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
    const storage Y3 = diff.get_lo();                              // < 2*mod
#else
    const storage t7 = fd.template mul<0>(W, Y1);     // t7 = W*Y1      < 2
    const storage t8 = fd.template mul<0>(M, t6);     // t8 = M*t6      < 2
    const storage Y3 = fd.template sub<2>(t8, t7);    // Y3 = t8-t7     < 2
#endif
    const storage ZZ3 = fd.template mul<0>(V, ZZ1);   // ZZ3 = V*ZZ1    < 2
    const storage ZZZ3 = fd.template mul<0>(W, ZZZ1); // ZZZ3 = W*ZZZ1  < 2
    return {X3, Y3, ZZ3, ZZZ3};
  }

  template <bool CHECK_ZERO = true> static __host__ __device__ __forceinline__ point_xyzz dbl(const point_xyzz &point, const FD &fd) {
    return dbl_2008_s_1<CHECK_ZERO>(point, fd);
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-add-2008-s
  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz add_2008_s(const point_xyzz &p1, const point_xyzz &p2, const FD &fd) {
    const storage X1 = p1.x;     // < 2
    const storage Y1 = p1.y;     // < 2
    const storage ZZ1 = p1.zz;   // < 2
    const storage ZZZ1 = p1.zzz; // < 2
    const storage X2 = p2.x;     // < 2
    const storage Y2 = p2.y;     // < 2
    const storage ZZ2 = p2.zz;   // < 2
    const storage ZZZ2 = p2.zzz; // < 2
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        return p2;
      if (unlikely(fd.is_zero(fd.reduce(ZZ2))))
        return p1;
    }
    const storage U1 = fd.template mul<0>(X1, ZZ2);  // U1 = X1*ZZ2   < 2
    const storage U2 = fd.template mul<0>(X2, ZZ1);  // U2 = X2*ZZ1   < 2
    const storage S1 = fd.template mul<0>(Y1, ZZZ2); // S1 = Y1*ZZZ2  < 2
    const storage S2 = fd.template mul<0>(Y2, ZZZ1); // S2 = Y2*ZZZ1  < 2
    const storage P = fd.template sub<2>(U2, U1);    // P = U2-U1     < 2
    const storage R = fd.template sub<2>(S2, S1);    // R = S2-S1     < 2
    if (CHECK_DOUBLE) {
      if (unlikely(fd.is_zero(fd.reduce(P))) && unlikely(fd.is_zero(fd.reduce(R))))
        return dbl<false>(p1, fd);
    }
    const storage PP = fd.template sqr<0>(P);       // PP = P^2        < 2
    const storage PPP = fd.template mul<0>(P, PP);  // PPP = P*PP      < 2
    const storage Q = fd.template mul<0>(U1, PP);   // Q = U1*PP       < 2
    const storage t0 = fd.template sqr<0>(R);       // t0 = R^2        < 2
    const storage t1 = fd.template dbl<2>(Q);       // t1 = 2*Q        < 2
    const storage t2 = fd.template sub<2>(t0, PPP); // t2 = t0-PPP     < 2
    const storage X3 = fd.template sub<2>(t2, t1);  // X3 = t2-t1      < 2
    const storage t3 = fd.template sub<2>(Q, X3);   // t3 = Q-X3       < 2
#ifdef __CUDA_ARCH__
    // Y3 = R*t3 - S1*PPP (requires R, t3, S1, PPP < 2*mod)
    const storage_wide t4_wide = fd.template mul_wide<0>(S1, PPP); // < 4*mod^2
    const storage_wide t5_wide = fd.template mul_wide<0>(R, t3);   // < 4*mod^2
    storage_wide diff = fd.template sub_wide<4>(t5_wide, t4_wide); // < 4*mod^2
    fd.redc_wide_inplace(diff);                                    // < 2*mod, hi limbs 0
    const storage Y3 = diff.get_lo();                              // < 2*mod
#else
    const storage t4 = fd.template mul<0>(S1, PPP);   // t4 = S1*PPP     < 2
    const storage t5 = fd.template mul<0>(R, t3);     // t5 = R*t3       < 2
    const storage Y3 = fd.template sub<2>(t5, t4);    // Y3 = t5-t4      < 2
#endif

    const storage t6 = fd.template mul<0>(ZZ2, PP);    // t6 = ZZ2*PP     < 2
    const storage ZZ3 = fd.template mul<0>(ZZ1, t6);   // ZZ3 = ZZ1*t6    < 2
    const storage t7 = fd.template mul<0>(ZZZ2, PPP);  // t7 = ZZZ2*PPP   < 2
    const storage ZZZ3 = fd.template mul<0>(ZZZ1, t7); // ZZZ3 = ZZZ1*t7  < 2
    return {X3, Y3, ZZ3, ZZZ3};
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz add(const point_xyzz &p1, const point_xyzz &p2, const FD &fd) {
    return add_2008_s<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
  }

  // https://hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz add_madd_2008_s(const point_xyzz &p1, const point_affine &p2, const FD &fd) {
    const storage X1 = p1.x;     // < 2
    const storage Y1 = p1.y;     // < 2
    const storage ZZ1 = p1.zz;   // < 2
    const storage ZZZ1 = p1.zzz; // < 2
    const storage X2 = p2.x;     // < 1
    const storage Y2 = p2.y;     // < 1
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(fd.reduce(ZZ1))))
        return point_affine::to_xyzz(p2, fd);
    }
    const storage U2 = fd.template mul<0>(X2, ZZ1);  // U2 = X2*ZZ1       < 2
    const storage S2 = fd.template mul<0>(Y2, ZZZ1); // S2 = Y2*ZZZ1      < 2
    const storage P = fd.template sub<2>(U2, X1);    // P = U2-X1         < 2
    const storage R = fd.template sub<2>(S2, Y1);    // R = S2-Y1         < 2
    if (CHECK_DOUBLE) {
      if (unlikely(fd.is_zero(fd.reduce(P))) && unlikely(fd.is_zero(fd.reduce(R))))
        return dbl(p2, fd);
    }
    const storage PP = fd.template sqr<0>(P);       // PP = P^2           < 2
    const storage PPP = fd.template mul<0>(P, PP);  // PPP = P*PP         < 2
    const storage Q = fd.template mul<0>(X1, PP);   // Q = X1*PP          < 2
    const storage t0 = fd.template sqr<0>(R);       // t0 = R^2           < 2
    const storage t1 = fd.template dbl<2>(Q);       // t1 = 2*Q           < 2
    const storage t2 = fd.template sub<2>(t0, PPP); // t2 = t0-PPP        < 2
    const storage X3 = fd.template sub<2>(t2, t1);  // X3 = t2-t1         < 2
    const storage t3 = fd.template sub<2>(Q, X3);   // t3 = Q-X3          < 2
#ifdef __CUDA_ARCH__
    // Y3 = R*t3-Y1*PPP
    const storage_wide t4_wide = fd.template mul_wide<0>(Y1, PPP); //         < 4*mod^2
    const storage_wide t5_wide = fd.template mul_wide<0>(R, t3);   //         < 4*mod^2
    storage_wide diff = fd.template sub_wide<4>(t5_wide, t4_wide); //         < 4*mod^2
    fd.redc_wide_inplace(diff);                                    //         < 2*mod, hi limbs 0
    const storage Y3 = diff.get_lo();                              //         < 2*mod
#else
    const storage t4 = fd.template mul<0>(Y1, PPP);   // t4 = Y1*PPP        < 2
    const storage t5 = fd.template mul<0>(R, t3);     // t5 = R*t3          < 2
    const storage Y3 = fd.template sub<2>(t5, t4);    // Y3 = t5-t4         < 2
#endif
    const storage ZZ3 = fd.template mul<0>(ZZ1, PP);    // ZZ3 = ZZ1*PP       < 2
    const storage ZZZ3 = fd.template mul<0>(ZZZ1, PPP); // ZZZ3 = ZZZ1*PPP    < 2
    return {X3, Y3, ZZ3, ZZZ3};
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz add(const point_xyzz &p1, const point_affine &p2, const FD &fd) {
    return add_madd_2008_s<CHECK_ZERO, CHECK_DOUBLE>(p1, p2, fd);
  }

  static __host__ __device__ __forceinline__ point_projective add(const point_projective &p1, const point_projective &p2, const FD &fd) {
    const storage X1 = p1.x;                                                 //                   < 2
    const storage Y1 = p1.y;                                                 //                   < 2
    const storage Z1 = p1.z;                                                 //                   < 2
    const storage X2 = p2.x;                                                 //                   < 2
    const storage Y2 = p2.y;                                                 //                   < 2
    const storage Z2 = p2.z;                                                 //                   < 2
    const storage t00 = fd.template mul<0>(X1, X2);                          // t00 ← X1 · X2     < 2
    const storage t01 = fd.template mul<0>(Y1, Y2);                          // t01 ← Y1 · Y2     < 2
    const storage t02 = fd.template mul<0>(Z1, Z2);                          // t02 ← Z1 · Z2     < 2
    const storage t03 = fd.template add<0>(X1, Y1);                          // t03 ← X1 + Y1     < 4
    const storage t04 = fd.template add<0>(X2, Y2);                          // t04 ← X2 + Y2     < 4
    const storage t05 = fd.template mul<2>(t03, t04);                        // t03 ← t03 · t04   < 3
    const storage t06 = fd.template add<0>(t00, t01);                        // t06 ← t00 + t01   < 4
    const storage t07 = fd.template reduce<2>(fd.template sub<4>(t05, t06)); // t05 ← t05 − t06   < 2
    const storage t08 = fd.template add<0>(Y1, Z1);                          // t08 ← Y1 + Z1     < 4
    const storage t09 = fd.template add<0>(Y2, Z2);                          // t09 ← Y2 + Z2     < 4
    const storage t10 = fd.template mul<2>(t08, t09);                        // t10 ← t08 · t09   < 3
    const storage t11 = fd.template add<0>(t01, t02);                        // t11 ← t01 + t02   < 4
    const storage t12 = fd.template reduce<2>(fd.template sub<4>(t10, t11)); // t12 ← t10 − t11   < 2
    const storage t13 = fd.template add<0>(X1, Z1);                          // t13 ← X1 + Z1     < 4
    const storage t14 = fd.template add<0>(X2, Z2);                          // t14 ← X2 + Z2     < 4
    const storage t15 = fd.template mul<2>(t13, t14);                        // t15 ← t13 · t14   < 3
    const storage t16 = fd.template add<0>(t00, t02);                        // t16 ← t00 + t02   < 4
    const storage t17 = fd.template reduce<2>(fd.template sub<4>(t15, t16)); // t17 ← t15 − t16   < 2
    const storage t18 = fd.template dbl<2>(t00);                             // t18 ← t00 + t00   < 2
    const storage t19 = fd.template add<2>(t18, t00);                        // t19 ← t18 + t00   < 2
    const storage t20 = fd.template mul<2>(3 * B_VALUE, t02);                // t20 ← b3 · t02    < 2
    const storage t21 = fd.template add<2>(t01, t20);                        // t21 ← t01 + t20   < 2
    const storage t22 = fd.template sub<2>(t01, t20);                        // t22 ← t01 − t20   < 2
    const storage t23 = fd.template mul<2>(3 * B_VALUE, t17);                // t23 ← b3 · t17    < 2
#ifdef __CUDA_ARCH__
    // X3 ← t07 · t22 - t12 · t23
    const storage_wide t24_wide = fd.template mul_wide<0>(t12, t23);         //                   < 4*mod^2
    const storage_wide t25_wide = fd.template mul_wide<0>(t07, t22);         //                   < 4*mod^2
    storage_wide t25mt24_wide = fd.template sub_wide<4>(t25_wide, t24_wide); //                   < 4*mod^2
    fd.redc_wide_inplace(t25mt24_wide);                                      //                   < 2*mod, hi limbs 0
    const storage X3 = t25mt24_wide.get_lo();                                //                   < 2*mod
    // Y3 ← t22 · t21 + t23 · t19
    const storage t21_red = fd.template reduce<1>(t21);                      //                   < 1*mod
    const storage t19_red = fd.template reduce<1>(t19);                      //                   < 1*mod
    const storage_wide t27_wide = fd.template mul_wide<0>(t23, t19_red);     //                   < 2*mod^2
    const storage_wide t28_wide = fd.template mul_wide<0>(t22, t21_red);     //                   < 2*mod^2
    storage_wide t28pt27_wide = fd.template add_wide<4>(t28_wide, t27_wide); //                   < 4*mod^2
    fd.redc_wide_inplace(t28pt27_wide);                                      //                   < 2*mod, hi limbs 0
    const storage Y3 = t28pt27_wide.get_lo();                                //                   < 2*mod
    // Z3 ← t21 · t12 + t19 · t07
    const storage_wide t30_wide = fd.template mul_wide<0>(t19_red, t07);     //                   < 2*mod^2
    const storage_wide t31_wide = fd.template mul_wide<0>(t21_red, t12);     //                   < 2*mod^2
    storage_wide t31pt30_wide = fd.template add_wide<4>(t31_wide, t30_wide); //                   < 4*mod^2
    fd.redc_wide_inplace(t31pt30_wide);                                      //                   < 2*mod, hi limbs 0
    const storage Z3 = t31pt30_wide.get_lo();                                //                   < 2*mod
#else
    const storage t24 = fd.template mul<0>(t12, t23); // t24 ← t12 · t23   < 2
    const storage t25 = fd.template mul<0>(t07, t22); // t25 ← t07 · t22   < 2
    const storage X3 = fd.template sub<2>(t25, t24);  // X3 ← t25 − t24    < 2
    const storage t27 = fd.template mul<0>(t23, t19); // t27 ← t23 · t19   < 2
    const storage t28 = fd.template mul<0>(t22, t21); // t28 ← t22 · t21   < 2
    const storage Y3 = fd.template add<2>(t28, t27);  // Y3 ← t28 + t27    < 2
    const storage t30 = fd.template mul<0>(t19, t07); // t30 ← t19 · t07   < 2
    const storage t31 = fd.template mul<0>(t21, t12); // t31 ← t21 · t12   < 2
    const storage Z3 = fd.template add<2>(t31, t30);  // Z3 ← t31 + t30    < 2
#endif
    return {X3, Y3, Z3};
  }

  // https://eprint.iacr.org/2015/1060.pdf
  static __host__ __device__ __forceinline__ point_projective add(const point_projective &p1, const point_affine &p2, const FD &fd) {
    const storage X1 = p1.x;
    const storage Y1 = p1.y;
    const storage Z1 = p1.z;
    const storage X2 = p2.x;
    const storage Y2 = p2.y;
    storage t0 = fd.template mul<0>(X1, X2);          // 1. t0 ← X1 · X2
    storage t1 = fd.template mul<0>(Y1, Y2);          // 2. t1 ← Y1 · Y2
    storage t3 = fd.template add<2>(X2, Y2);          // 3. t3 ← X2 + Y2
    storage t4 = fd.template add<2>(X1, Y1);          // 4. t4 ← X1 + Y1
    t3 = fd.template mul<0>(t3, t4);                  // 5. t3 ← t3 · t4
    t4 = fd.template add<2>(t0, t1);                  // 6. t4 ← t0 + t1
    t3 = fd.template sub<2>(t3, t4);                  // 7. t3 ← t3 − t4
    t4 = fd.template mul<0>(Y2, Z1);                  // 8. t4 ← Y2 · Z1
    t4 = fd.template add<2>(t4, Y1);                  // 9. t4 ← t4 + Y1
    storage Y3 = fd.template mul<0>(X2, Z1);          // 10. Y3 ← X2 · Z1
    Y3 = fd.template add<2>(Y3, X1);                  // 11. Y3 ← Y3 + X1
    storage X3 = fd.template dbl<2>(t0);              // 12. X3 ← t0 + t0
    t0 = fd.template add<2>(X3, t0);                  // 13. t0 ← X3 + t0
    storage t2 = fd.template mul<2>(3 * B_VALUE, Z1); // 14. t2 ← b3 · Z1
    storage Z3 = fd.template add<2>(t1, t2);          // 15. Z3 ← t1 + t2
    t1 = fd.template sub<2>(t1, t2);                  // 16. t1 ← t1 − t2
    Y3 = fd.template mul<2>(3 * B_VALUE, Y3);         // 17. Y3 ← b3 · Y3
    X3 = fd.template mul<0>(t4, Y3);                  // 18. X3 ← t4 · Y3
    t2 = fd.template mul<0>(t3, t1);                  // 19. t2 ← t3 · t1
    X3 = fd.template sub<2>(t2, X3);                  // 20. X3 ← t2 − X3
    Y3 = fd.template mul<0>(Y3, t0);                  // 21. Y3 ← Y3 · t0
    t1 = fd.template mul<0>(t1, Z3);                  // 22. t1 ← t1 · Z3
    Y3 = fd.template add<2>(t1, Y3);                  // 23. Y3 ← t1 + Y3
    t0 = fd.template mul<0>(t0, t3);                  // 24. t0 ← t0 · t3
    Z3 = fd.template mul<0>(Z3, t4);                  // 25. Z3 ← Z3 · t4
    Z3 = fd.template add<2>(Z3, t0);                  // 26. Z3 ← Z3 + t0
    return {X3, Y3, Z3};
  }

  // https://eprint.iacr.org/2015/1060.pdf
  static __host__ __device__ __forceinline__ point_projective dbl(const point_projective &point, const FD &fd) {
    const storage X = point.x;
    const storage Y = point.y;
    const storage Z = point.z;
    storage t0 = fd.template sqr<0>(Y);       // 1. t0 ← Y · Y
    storage Z3 = fd.template dbl<2>(t0);      // 2. Z3 ← t0 + t0
    Z3 = fd.template dbl<2>(Z3);              // 3. Z3 ← Z3 + Z3
    Z3 = fd.template dbl<2>(Z3);              // 4. Z3 ← Z3 + Z3
    storage t1 = fd.template mul<0>(Y, Z);    // 5. t1 ← Y · Z
    storage t2 = fd.template sqr<0>(Z);       // 6. t2 ← Z · Z
    t2 = fd.template mul<2>(3 * B_VALUE, t2); // 7. t2 ← b3 · t2
    storage X3 = fd.template mul<0>(t2, Z3);  // 8. X3 ← t2 · Z3
    storage Y3 = fd.template add<2>(t0, t2);  // 9. Y3 ← t0 + t2
    Z3 = fd.template mul<0>(t1, Z3);          // 10. Z3 ← t1 · Z3
    t1 = fd.template dbl<2>(t2);              // 11. t1 ← t2 + t2
    t2 = fd.template add<2>(t1, t2);          // 12. t2 ← t1 + t2
    t0 = fd.template sub<2>(t0, t2);          // 13. t0 ← t0 − t2
    Y3 = fd.template mul<0>(t0, Y3);          // 14. Y3 ← t0 · Y3
    Y3 = fd.template add<2>(X3, Y3);          // 15. Y3 ← X3 + Y3
    t1 = fd.template mul<0>(X, Y);            // 16. t1 ← X · Y
    X3 = fd.template mul<0>(t0, t1);          // 17. X3 ← t0 · t1
    X3 = fd.template dbl<2>(X3);              // 18. X3 ← X3 + X3
    return {X3, Y3, Z3};
  }

  template <class FD_SCALAR>
  static __host__ __device__ __forceinline__ point_projective mul(const typename FD_SCALAR::storage &scalar, const point_projective &point, const FD &fd) {
    point_projective result = point_projective::point_at_infinity(fd);
    unsigned count = FD_SCALAR::TLC;
    while (count != 0 && scalar.limbs[count - 1] == 0)
      count--;
    point_projective temp = point;
    bool is_zero = true;
    for (unsigned i = 0; i < count; i++) {
      uint32_t limb = scalar.limbs[i];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned j = 0; j < 32; j++) {
        if (limb & 1) {
          result = is_zero ? temp : add(result, temp, fd);
          is_zero = false;
        }
        limb >>= 1;
        if (i == count - 1 && limb == 0)
          break;
        temp = dbl(temp, fd);
      }
    }
    return result;
  }

  template <class FD_SCALAR, bool CHECK_ZERO = true>
  static __host__ __device__ __forceinline__ point_jacobian mul(const typename FD_SCALAR::storage &scalar, const point_jacobian &point, const FD &fd) {
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(point.z)))
        return point_jacobian::point_at_infinity(fd);
    }
    point_jacobian result = point_jacobian::point_at_infinity(fd);
    unsigned count = FD_SCALAR::TLC;
    while (count != 0 && scalar.limbs[count - 1] == 0)
      count--;
    point_jacobian temp = point;
    bool is_zero = true;
    for (unsigned i = 0; i < count; i++) {
      uint32_t limb = scalar.limbs[i];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned j = 0; j < 32; j++) {
        if (limb & 1) {
          result = is_zero ? temp : add<false>(result, temp, fd);
          is_zero = false;
        }
        limb >>= 1;
        if (i == count - 1 && limb == 0)
          break;
        temp = dbl<false>(temp, fd);
      }
    }
    return result;
  }

  template <class FD_SCALAR, bool CHECK_ZERO = true>
  static __host__ __device__ __forceinline__ point_xyzz mul(const typename FD_SCALAR::storage &scalar, const point_xyzz &point, const FD &fd) {
    if (CHECK_ZERO) {
      if (unlikely(fd.is_zero(point.zz)))
        return point_xyzz::point_at_infinity(fd);
    }
    point_xyzz result = point_xyzz::point_at_infinity(fd);
    unsigned count = FD_SCALAR::TLC;
    while (count != 0 && scalar.limbs[count - 1] == 0)
      count--;
    point_xyzz temp = point;
    bool is_zero = true;
    for (unsigned i = 0; i < count; i++) {
      uint32_t limb = scalar.limbs[i];
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
      for (unsigned j = 0; j < 32; j++) {
        if (limb & 1) {
          result = is_zero ? temp : add<false>(result, temp, fd);
          is_zero = false;
        }
        limb >>= 1;
        if (i == count - 1 && limb == 0)
          break;
        temp = dbl<false>(temp, fd);
      }
    }
    return result;
  }

  static __host__ __device__ __forceinline__ point_projective sub(const point_projective &p1, const point_affine &p2, const FD &fd) {
    return add(p1, point_affine::neg(p2, fd), fd);
  }

  static __host__ __device__ __forceinline__ point_projective sub(const point_projective &p1, const point_projective &p2, const FD &fd) {
    return add(p1, point_projective::neg(p2, fd), fd);
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian sub(const point_jacobian &p1, const point_affine &p2, const FD &fd) {
    return add<CHECK_ZERO, CHECK_DOUBLE>(p1, point_affine::neg(p2, fd), fd);
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_jacobian sub(const point_jacobian &p1, const point_jacobian &p2, const FD &fd) {
    return add<CHECK_ZERO, CHECK_DOUBLE>(p1, point_jacobian::neg(p2, fd), fd);
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz sub(const point_xyzz &p1, const point_affine &p2, const FD &fd) {
    return add<CHECK_ZERO, CHECK_DOUBLE>(p1, point_affine::neg(p2, fd), fd);
  }

  template <bool CHECK_ZERO = true, bool CHECK_DOUBLE = true>
  static __host__ __device__ __forceinline__ point_xyzz sub(const point_xyzz &p1, const point_xyzz &p2, const FD &fd) {
    return add<CHECK_ZERO, CHECK_DOUBLE>(p1, point_xyzz::neg(p2, fd), fd);
  }
};

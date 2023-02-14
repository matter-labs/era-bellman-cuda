#pragma once

#include "ec.cuh"
#include "ff_dispatch_st.cuh"

typedef ec<fd_p, 3> curve;
typedef curve::storage storage;
typedef curve::point_affine pa;
typedef curve::point_projective pp;
typedef curve::point_xyzz pz;

template <class POINT1, class POINT2 = POINT1> cudaError_t ec_add(const POINT1 *x, const POINT2 *y, POINT1 *result, unsigned count);
template <class POINT1, class POINT2 = POINT1> cudaError_t ec_sub(const POINT1 *x, const POINT2 *y, POINT1 *result, unsigned count);
template <class POINT> cudaError_t ec_dbl(const POINT *, POINT *, unsigned);
template <class POINT> cudaError_t ec_mul(const fd_q::storage *, const POINT *, POINT *, unsigned);

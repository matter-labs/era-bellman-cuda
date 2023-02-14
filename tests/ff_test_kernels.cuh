#pragma once

template <class FD> cudaError_t fields_add(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_sub(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_neg(const typename FD::storage *x, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_to_montgomery(const typename FD::storage *x, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_from_montgomery(const typename FD::storage *x, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_mul(const typename FD::storage *x, const typename FD::storage *y, typename FD::storage *result, unsigned count);

template <class FD> cudaError_t fields_dbl(const typename FD::storage *x, typename FD::storage *result, unsigned count);

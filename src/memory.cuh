#pragma once

#include <sm_32_intrinsics.h>

namespace memory {

enum class ld_modifier { none, g, cg, ca, cs, lu, cv };

template <typename T, ld_modifier MODIFIER> static constexpr __device__ __forceinline__ T ld_single(const T *ptr) {
  switch (MODIFIER) {
  case ld_modifier::none:
    return *ptr;
  case ld_modifier::g:
    return __ldg(ptr);
  case ld_modifier::cg:
    return __ldcg(ptr);
  case ld_modifier::ca:
    return __ldca(ptr);
  case ld_modifier::cs:
    return __ldcs(ptr);
  case ld_modifier::lu:
    return __ldlu(ptr);
  case ld_modifier::cv:
    return __ldcv(ptr);
  }
}

enum class st_modifier { none, wb, cg, cs, wt };

template <typename T, st_modifier MODIFIER> static constexpr __device__ __forceinline__ void st_single(T *ptr, T value) {
  switch (MODIFIER) {
  case st_modifier::none:
    *ptr = value;
    break;
  case st_modifier::wb:
    __stwb(ptr, value);
    break;
  case st_modifier::cg:
    __stcg(ptr, value);
    break;
  case st_modifier::cs:
    __stcs(ptr, value);
    break;
  case st_modifier::wt:
    __stwt(ptr, value);
    break;
  }
}

template <typename T> __device__ __forceinline__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <unsigned STRIDE> __device__ __forceinline__ unsigned swap_index(unsigned index) {
  const unsigned i1 = index % STRIDE;
  const unsigned i2 = index / STRIDE;
  const unsigned i3 = i2 * STRIDE * 2;
  return i3 + i1;
}

template <typename T> __device__ __forceinline__ T shfl_xor(unsigned mask, T var, int laneMask, int width = warpSize) {
  return __shfl_xor_sync(mask, var, laneMask, width);
}

template <> __device__ __forceinline__ uint2 shfl_xor(unsigned mask, uint2 var, int laneMask, int width) {
  uint2 result;
  result.x = __shfl_xor_sync(mask, var.x, laneMask, width);
  result.y = __shfl_xor_sync(mask, var.y, laneMask, width);
  return result;
}

template <> __device__ __forceinline__ uint4 shfl_xor(unsigned mask, uint4 var, int laneMask, int width) {
  uint4 result;
  result.x = __shfl_xor_sync(mask, var.x, laneMask, width);
  result.y = __shfl_xor_sync(mask, var.y, laneMask, width);
  result.z = __shfl_xor_sync(mask, var.z, laneMask, width);
  result.w = __shfl_xor_sync(mask, var.w, laneMask, width);
  return result;
}

template <typename T, unsigned STRIDE_ROW, unsigned COUNT_ROW, unsigned STRIDE_COL = STRIDE_ROW, unsigned COUNT_COL = COUNT_ROW>
__device__ __forceinline__ void transpose_tile(unsigned mask, T *u, const unsigned lane_id) {
  const bool swap_rows = !(lane_id & STRIDE_ROW);
  if (swap_rows) {
#pragma unroll
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < COUNT_COL; i++) {
    const unsigned index = swap_index<STRIDE_COL>(i);
    u[index] = shfl_xor(mask, u[index], STRIDE_COL);
  }
  if (swap_rows) {
#pragma unroll
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
}

template <class T, typename U, ld_modifier MODIFIER, unsigned STRIDE>
static constexpr __device__ __forceinline__ T ld(const T *address, const unsigned offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  T result = {};
  auto pa = reinterpret_cast<const U *>(address) + offset;
  auto pr = reinterpret_cast<U *>(&result);
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const auto pai = pa + i * STRIDE;
    const auto pri = pr + i;
    *pri = ld_single<U, MODIFIER>(pai);
  }
  return result;
}

template <class T, typename U, st_modifier MODIFIER, unsigned STRIDE>
static constexpr __device__ __forceinline__ void st(T *address, const T &value, const unsigned offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  auto pa = reinterpret_cast<U *>(address) + offset;
  auto pv = reinterpret_cast<const U *>(&value);
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    auto pai = pa + i * STRIDE;
    auto pvi = pv + i;
    st_single<U, MODIFIER>(pai, *pvi);
  }
}

template <typename U, unsigned STRIDE_COL> __device__ __forceinline__ void transpose_tile(const unsigned stride_row, U *tile, const unsigned lane_id) {
  switch (stride_row) {
  case 0:
    transpose_tile<U, 1, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 1:
    transpose_tile<U, 2, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 2:
    transpose_tile<U, 4, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 3:
    transpose_tile<U, 8, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 4:
    transpose_tile<U, 16, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  default:
    break;
  }
}

template <class T, typename U, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER, bool CHECK_INACTIVE>
__device__ __forceinline__ T ld_warp(const T *address, const unsigned offset, const unsigned lane_id) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % (sizeof(U) << LOG_WARP_SIZE) == 0);
  constexpr size_t count = sizeof(T) / (sizeof(U) << LOG_WARP_SIZE);
  constexpr unsigned threads_count = 1 << LOG_WARP_SIZE;
  const unsigned l = lane_id & (threads_count - 1);
  T result = {};
  auto pr = reinterpret_cast<U *>(&result);
#pragma unroll
  for (int i = 0; i < threads_count; i++) {
    const unsigned o = __shfl_sync(UINT32_MAX, offset, i, threads_count);
    if (CHECK_INACTIVE && o == UINT32_MAX)
      continue;
    const U *ap = reinterpret_cast<const U *>(address + o) + l;
#pragma unroll
    for (unsigned j = 0; j < count; j++) {
      const unsigned shift = j << LOG_WARP_SIZE;
      pr[i + shift] = ld_single<U, MODIFIER>(ap + shift);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const unsigned shift = i << LOG_WARP_SIZE;
    U *tile = pr + shift;
    constexpr unsigned stride = threads_count >> 1;
#pragma unroll
    for (unsigned j = 0; j < LOG_WARP_SIZE; j++)
      transpose_tile<U, stride>(j, tile, l);
  }
  return result;
}

template <class T, typename U, unsigned LOG_WARP_SIZE, st_modifier MODIFIER, bool CHECK_INACTIVE>
__device__ __forceinline__ void st_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % (sizeof(U) << LOG_WARP_SIZE) == 0);
  constexpr size_t count = sizeof(T) / (sizeof(U) << LOG_WARP_SIZE);
  constexpr unsigned threads_count = 1 << LOG_WARP_SIZE;
  const unsigned l = lane_id & (threads_count - 1);
  T value_copy = value;
  auto pv = reinterpret_cast<U *>(&value_copy);
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const unsigned shift = i << LOG_WARP_SIZE;
    U *tile = pv + shift;
    constexpr unsigned stride = threads_count >> 1;
#pragma unroll
    for (int j = LOG_WARP_SIZE - 1; j >= 0; j--)
      transpose_tile<U, stride>(j, tile, l);
  }
#pragma unroll
  for (int i = 0; i < threads_count; i++) {
    const unsigned o = __shfl_sync(UINT32_MAX, offset, i, threads_count);
    if (CHECK_INACTIVE && o == UINT32_MAX)
      continue;
    U *ap = reinterpret_cast<U *>(address + o) + l;
#pragma unroll
    for (unsigned j = 0; j < count; j++) {
      const unsigned shift = j << LOG_WARP_SIZE;
      st_single<U, MODIFIER>(ap + shift, pv[i + shift]);
    }
  }
}

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
static constexpr __device__ __forceinline__ T load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint4 _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
};

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) == 0, uint4>>
static constexpr __device__ __forceinline__ T load_warp(const T *address, const unsigned offset, const unsigned lane_id, [[maybe_unused]] uint4 _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
};

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1,
          typename U = enable_if_t<(sizeof(T) % sizeof(uint4) != 0) && (sizeof(T) % sizeof(uint2) == 0), uint2>>
static constexpr __device__ __forceinline__ T load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint2 _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
};

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) != 0 && sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) == 0, uint2>>
static constexpr __device__ __forceinline__ T load_warp(const T *address, const unsigned offset, const unsigned lane_id, [[maybe_unused]] uint2 _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
};

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint2) != 0, unsigned>>
static constexpr __device__ __forceinline__ T load(const T *address, const unsigned offset = 0, [[maybe_unused]] unsigned _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
};

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) != 0, unsigned>>
static constexpr __device__ __forceinline__ T load_warp(const T *address, const unsigned offset, const unsigned lane_id,
                                                        [[maybe_unused]] unsigned _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
};

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
static constexpr __device__ __forceinline__ void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] uint4 _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) == 0, uint4>>
static constexpr __device__ __forceinline__ void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                            [[maybe_unused]] uint4 _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1,
          typename U = enable_if_t<(sizeof(T) % sizeof(uint4) != 0) && (sizeof(T) % sizeof(uint2) == 0), uint2>>
static constexpr __device__ __forceinline__ void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] uint2 _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) != 0 && sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) == 0, uint2>>
static constexpr __device__ __forceinline__ void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                            [[maybe_unused]] uint2 _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint2) != 0, unsigned>>
static constexpr __device__ __forceinline__ void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] unsigned _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) != 0, unsigned>>
static constexpr __device__ __forceinline__ void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                            [[maybe_unused]] unsigned _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

} // namespace memory
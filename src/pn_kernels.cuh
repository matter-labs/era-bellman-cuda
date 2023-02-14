#pragma once

#include "ff_dispatch_st.cuh"

namespace pn {

cudaError_t set_up_powers();

template <unsigned COL_COUNT> cudaError_t transpose(unsigned *dst, const unsigned *src, unsigned log_rows_count, cudaStream_t stream);

cudaError_t fill_transposed_range(unsigned *values, unsigned columns_count, unsigned log_rows_count, cudaStream_t stream);

cudaError_t mark_ends_of_runs(unsigned *values, const unsigned *run_lengths, const unsigned *run_offsets, unsigned count, cudaStream_t stream);

cudaError_t generate_permutation_matrix(fd_q::storage *values, const fd_q::storage *scalars, const unsigned *cell_indexes, const unsigned *run_indexes,
                                        const unsigned *run_lengths, const unsigned *run_offsets, unsigned columns_count, unsigned log_rows_count,
                                        cudaStream_t stream);

cudaError_t set_values_from_packed_bits(fd_q::storage *values, const unsigned *packet_bits, unsigned count, cudaStream_t stream);

} // namespace pn
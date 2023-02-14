#include "allocator.cuh"
#include "bellman-cuda-cub.cuh"
#include "common.cuh"
#include "msm.cuh"
#include <algorithm>

namespace msm {

using namespace allocator;
using namespace common;

unsigned get_window_bits_count(const unsigned log_scalars_count) {
  switch (log_scalars_count) {
  case 14:
    return 13;
  case 15:
  case 16:
  case 17:
  case 18:
  case 19:
    return 15;
  case 20:
  case 21:
    return 16;
  case 22:
    return 17;
  case 23:
    return 19;
  case 24:
  case 25:
    return 20;
  case 26:
    return 22;
  default:
    return max(log_scalars_count, 3u);
  }
}

unsigned get_log_min_inputs_count(const unsigned log_scalars_count) {
  switch (log_scalars_count) {
  case 18:
  case 19:
    return 17;
  case 20:
    return 18;
  case 21:
  case 22:
  case 23:
  case 24:
    return 19;
  case 25:
  case 26:
    return 22;
  default:
    return log_scalars_count;
  }
}

unsigned get_windows_count(const unsigned window_bits_count) { return (fd_q::MBC - 1) / window_bits_count + 1; }

unsigned get_optimal_log_data_split(const unsigned mpc, const unsigned source_window_bits, const unsigned target_window_bits,
                                    const unsigned target_windows_count) {
#define MAX_THREADS 32
#define MIN_BLOCKS 16
  const unsigned full_occupancy = mpc * MAX_THREADS * MIN_BLOCKS;
  const unsigned target = full_occupancy << 6;
  const unsigned unit_threads_count = target_windows_count << target_window_bits;
  const unsigned split_target = log2_ceiling(target / unit_threads_count);
  const unsigned split_limit = source_window_bits - target_window_bits - 1;
  return std::min(split_target, split_limit);
}

cudaError_t set_up() { return set_kernel_attributes(); }

struct extended_configuration {
  execution_configuration exec_cfg;
  cudaPointerAttributes scalars_attributes;
  cudaPointerAttributes bases_attributes;
  cudaPointerAttributes results_attributes;
  unsigned log_min_inputs_count;
  unsigned log_max_inputs_count;
  cudaDeviceProp device_props;
};

cudaError_t schedule_execution(const extended_configuration &cfg, const bool dry_run) {
  typedef allocation<point_xyzz> buckets;
  typedef allocation<fd_q::storage> scalars;
  typedef allocation<point_affine> bases;
  typedef allocation<unsigned> unsigned_ints;
  typedef allocation<void> temp_storage;
  typedef allocation<point_jacobian> buckets_jacobian;
  execution_configuration ec = cfg.exec_cfg;
  cudaMemPool_t pool = ec.mem_pool;
  cudaStream_t stream = ec.stream;
  const unsigned log_scalars_count = ec.log_scalars_count;
  const unsigned scalars_count = 1 << log_scalars_count;
  const unsigned log_min_inputs_count = cfg.log_min_inputs_count;
  const unsigned log_max_inputs_count = cfg.log_max_inputs_count;
  const unsigned bits_count_pass_one = get_window_bits_count(log_scalars_count);
  const unsigned signed_bits_count_pass_one = bits_count_pass_one - 1;
  const unsigned windows_count_pass_one = get_windows_count(bits_count_pass_one);
  const unsigned buckets_count_pass_one = windows_count_pass_one << signed_bits_count_pass_one;
  const unsigned top_window_unused_bits = windows_count_pass_one * bits_count_pass_one - fd_q::MBC;
  const unsigned extended_buckets_count_pass_one = buckets_count_pass_one + windows_count_pass_one - 1 + (1 << top_window_unused_bits);
  bool copy_scalars = cfg.scalars_attributes.type == cudaMemoryTypeUnregistered || cfg.scalars_attributes.type == cudaMemoryTypeHost;
  bool copy_bases = cfg.bases_attributes.type == cudaMemoryTypeUnregistered || cfg.bases_attributes.type == cudaMemoryTypeHost;
  bool copy_results = cfg.results_attributes.type == cudaMemoryTypeUnregistered || cfg.results_attributes.type == cudaMemoryTypeHost;

  cudaEvent_t execution_started_event;
  if (!dry_run) {
    HANDLE_CUDA_ERROR(cudaEventCreate(&execution_started_event, cudaEventDisableTiming));
    HANDLE_CUDA_ERROR(cudaEventRecord(execution_started_event, stream));
  }

  cudaStream_t stream_copy_scalars;
  cudaStream_t stream_copy_bases;
  cudaStream_t stream_copy_finished;
  cudaStream_t stream_sort_a;
  cudaStream_t stream_sort_b;
  if (!dry_run) {
    if (copy_scalars) {
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_copy_scalars));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_scalars, execution_started_event));
    }
    if (copy_bases) {
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_copy_bases));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_bases, execution_started_event));
    }
    if (copy_scalars || copy_bases) {
      HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_copy_finished));
    }
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_sort_a));
    HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_sort_a, execution_started_event));
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream_sort_b));
    HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_sort_b, execution_started_event));
    HANDLE_CUDA_ERROR(cudaEventDestroy(execution_started_event));
  }

  buckets buckets_pass_one;
  HANDLE_CUDA_ERROR(allocate(buckets_pass_one, buckets_count_pass_one, pool, stream));

  scalars inputs_scalars;
  cudaEvent_t event_scalars_free;
  cudaEvent_t event_scalars_loaded;
  if (copy_scalars) {
    HANDLE_CUDA_ERROR(allocate(inputs_scalars, 1 << cfg.log_max_inputs_count, pool, stream));
    HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_scalars_free, cudaEventDisableTiming));
    HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_scalars_loaded, cudaEventDisableTiming));
  }

  bases inputs_bases;
  cudaEvent_t event_bases_free;
  cudaEvent_t event_bases_loaded;
  if (copy_bases) {
    HANDLE_CUDA_ERROR(allocate(inputs_bases, 1 << cfg.log_max_inputs_count, pool, stream));
    HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_bases_free, cudaEventDisableTiming));
    HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_bases_loaded, cudaEventDisableTiming));
  }

  for (unsigned inputs_offset = 0, log_inputs_count = log_min_inputs_count; inputs_offset < scalars_count;) {
    const unsigned inputs_count = 1 << log_inputs_count;
    bool is_first_loop = inputs_offset == 0;
    bool is_last_loop = inputs_offset + inputs_count == scalars_count;
    const unsigned input_indexes_count = windows_count_pass_one << log_inputs_count;

    if (!dry_run) {
      if (is_first_loop && (cfg.scalars_attributes.type == cudaMemoryTypeUnregistered || cfg.bases_attributes.type == cudaMemoryTypeUnregistered))
        HANDLE_CUDA_ERROR(initialize_buckets(buckets_pass_one, buckets_count_pass_one, stream));
      if (copy_scalars) {
        HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_scalars, event_scalars_free));
        const size_t inputs_size = sizeof(fd_q::storage) << log_inputs_count;
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(inputs_scalars, ec.scalars + inputs_offset, inputs_size, cudaMemcpyHostToDevice, stream_copy_scalars));
        HANDLE_CUDA_ERROR(cudaEventRecord(event_scalars_loaded, stream_copy_scalars));
      }
      if (copy_bases) {
        if (copy_scalars)
          HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_bases, event_scalars_loaded));
        HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_bases, event_bases_free));
        const size_t bases_size = sizeof(point_affine) << log_inputs_count;
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(inputs_bases, ec.bases + inputs_offset, bases_size, cudaMemcpyHostToDevice, stream_copy_bases));
        HANDLE_CUDA_ERROR(cudaEventRecord(event_bases_loaded, stream_copy_bases));
      }
      if (is_last_loop && (copy_bases || copy_scalars)) {
        if (copy_scalars)
          HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_finished, event_scalars_loaded));
        if (copy_bases)
          HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_copy_finished, event_bases_loaded));
        if (ec.h2d_copy_finished)
          HANDLE_CUDA_ERROR(cudaEventRecord(ec.h2d_copy_finished, stream_copy_finished));
        if (ec.h2d_copy_finished_callback)
          HANDLE_CUDA_ERROR(cudaLaunchHostFunc(stream_copy_finished, ec.h2d_copy_finished_callback, ec.h2d_copy_finished_callback_data));
      }
      if (is_first_loop && cfg.scalars_attributes.type != cudaMemoryTypeUnregistered && cfg.bases_attributes.type != cudaMemoryTypeUnregistered)
        HANDLE_CUDA_ERROR(initialize_buckets(buckets_pass_one, buckets_count_pass_one, stream));
    }

    // compute bucket indexes
    unsigned_ints bucket_indexes;
    HANDLE_CUDA_ERROR(allocate(bucket_indexes, input_indexes_count + inputs_count, pool, stream));
    unsigned_ints base_indexes;
    HANDLE_CUDA_ERROR(allocate(base_indexes, input_indexes_count + inputs_count, pool, stream));
    if (!dry_run) {
      if (copy_scalars)
        HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream, event_scalars_loaded));
      HANDLE_CUDA_ERROR(compute_bucket_indexes(copy_scalars ? inputs_scalars : ec.scalars + inputs_offset, windows_count_pass_one, bits_count_pass_one,
                                               bucket_indexes + inputs_count, base_indexes + inputs_count, inputs_count, stream));
      if (copy_scalars)
        HANDLE_CUDA_ERROR(cudaEventRecord(event_scalars_free, stream));
    }

    if (is_last_loop && copy_scalars)
      HANDLE_CUDA_ERROR(free(inputs_scalars, stream));

    // sort base indexes by bucket indexes
    temp_storage input_indexes_sort_temp_storage;
    size_t input_indexes_sort_temp_storage_bytes = 0;
    HANDLE_CUDA_ERROR(sort_pairs(input_indexes_sort_temp_storage, input_indexes_sort_temp_storage_bytes, bucket_indexes + inputs_count, bucket_indexes,
                                 base_indexes + inputs_count, base_indexes, inputs_count, 0, bits_count_pass_one));
    HANDLE_CUDA_ERROR(allocate(input_indexes_sort_temp_storage, input_indexes_sort_temp_storage_bytes, pool, stream));
    if (!dry_run)
      for (unsigned i = 0; i < windows_count_pass_one; i++) {
        unsigned offset_out = i * inputs_count;
        unsigned offset_in = offset_out + inputs_count;
        HANDLE_CUDA_ERROR(sort_pairs(input_indexes_sort_temp_storage, input_indexes_sort_temp_storage_bytes, bucket_indexes + offset_in,
                                     bucket_indexes + offset_out, base_indexes + offset_in, base_indexes + offset_out, inputs_count, 0, bits_count_pass_one,
                                     stream));
      }
    HANDLE_CUDA_ERROR(free(input_indexes_sort_temp_storage, stream));

    // run length encode bucket runs
    unsigned_ints unique_bucket_indexes;
    HANDLE_CUDA_ERROR(allocate(unique_bucket_indexes, extended_buckets_count_pass_one, pool, stream));
    unsigned_ints bucket_run_lengths;
    HANDLE_CUDA_ERROR(allocate(bucket_run_lengths, extended_buckets_count_pass_one, pool, stream));
    unsigned_ints bucket_runs_count;
    HANDLE_CUDA_ERROR(allocate(bucket_runs_count, 1, pool, stream));
    temp_storage encode_temp_storage;
    size_t encode_temp_storage_bytes = 0;
    HANDLE_CUDA_ERROR(run_length_encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indexes, unique_bucket_indexes, bucket_run_lengths,
                                        bucket_runs_count, input_indexes_count));
    HANDLE_CUDA_ERROR(allocate(encode_temp_storage, encode_temp_storage_bytes, pool, stream));
    if (!dry_run)
      HANDLE_CUDA_ERROR(run_length_encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indexes, unique_bucket_indexes, bucket_run_lengths,
                                          bucket_runs_count, input_indexes_count, stream));
    HANDLE_CUDA_ERROR(free(encode_temp_storage, stream));
    HANDLE_CUDA_ERROR(free(bucket_indexes, stream));

    // compute bucket run offsets
    unsigned_ints bucket_run_offsets;
    HANDLE_CUDA_ERROR(allocate(bucket_run_offsets, extended_buckets_count_pass_one, pool, stream));
    temp_storage scan_temp_storage;
    size_t scan_temp_storage_bytes = 0;
    HANDLE_CUDA_ERROR(exclusive_sum(scan_temp_storage, scan_temp_storage_bytes, bucket_run_lengths, bucket_run_offsets, extended_buckets_count_pass_one));
    HANDLE_CUDA_ERROR(allocate(scan_temp_storage, scan_temp_storage_bytes, pool, stream));
    if (!dry_run)
      HANDLE_CUDA_ERROR(
          exclusive_sum(scan_temp_storage, scan_temp_storage_bytes, bucket_run_lengths, bucket_run_offsets, extended_buckets_count_pass_one, stream));
    HANDLE_CUDA_ERROR(free(scan_temp_storage, stream));

    if (!dry_run)
      HANDLE_CUDA_ERROR(remove_zero_buckets(unique_bucket_indexes, bucket_run_lengths, bucket_runs_count, extended_buckets_count_pass_one, stream));
    HANDLE_CUDA_ERROR(free(bucket_runs_count, stream));

    // sort run offsets by run lengths
    // sort run indexes by run lengths
    unsigned_ints sorted_bucket_run_lengths;
    HANDLE_CUDA_ERROR(allocate(sorted_bucket_run_lengths, extended_buckets_count_pass_one, pool, stream));
    unsigned_ints sorted_bucket_run_offsets;
    HANDLE_CUDA_ERROR(allocate(sorted_bucket_run_offsets, extended_buckets_count_pass_one, pool, stream));
    unsigned_ints sorted_unique_bucket_indexes;
    HANDLE_CUDA_ERROR(allocate(sorted_unique_bucket_indexes, extended_buckets_count_pass_one, pool, stream));

    temp_storage sort_offsets_temp_storage;
    size_t sort_offsets_temp_storage_bytes = 0;
    HANDLE_CUDA_ERROR(sort_pairs_descending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_run_lengths, sorted_bucket_run_lengths,
                                            bucket_run_offsets, sorted_bucket_run_offsets, extended_buckets_count_pass_one, 0, log_inputs_count + 1));
    HANDLE_CUDA_ERROR(allocate(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, pool, stream));

    temp_storage sort_indexes_temp_storage;
    size_t sort_indexes_temp_storage_bytes = 0;
    HANDLE_CUDA_ERROR(sort_pairs_descending(sort_indexes_temp_storage, sort_indexes_temp_storage_bytes, bucket_run_lengths, sorted_bucket_run_lengths,
                                            unique_bucket_indexes, sorted_unique_bucket_indexes, extended_buckets_count_pass_one, 0, log_inputs_count + 1));
    HANDLE_CUDA_ERROR(allocate(sort_indexes_temp_storage, sort_indexes_temp_storage_bytes, pool, stream));

    if (!dry_run) {
      cudaEvent_t event_sort_inputs_ready;
      HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_sort_inputs_ready, cudaEventDisableTiming));
      HANDLE_CUDA_ERROR(cudaEventRecord(event_sort_inputs_ready, stream));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_sort_a, event_sort_inputs_ready));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream_sort_b, event_sort_inputs_ready));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_sort_inputs_ready));
      HANDLE_CUDA_ERROR(sort_pairs_descending(sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, bucket_run_lengths, sorted_bucket_run_lengths,
                                              bucket_run_offsets, sorted_bucket_run_offsets, extended_buckets_count_pass_one, 0, log_inputs_count + 1,
                                              stream_sort_a));
      HANDLE_CUDA_ERROR(sort_pairs_descending(sort_indexes_temp_storage, sort_indexes_temp_storage_bytes, bucket_run_lengths, sorted_bucket_run_lengths,
                                              unique_bucket_indexes, sorted_unique_bucket_indexes, extended_buckets_count_pass_one, 0, log_inputs_count + 1,
                                              stream_sort_b));
      cudaEvent_t event_sort_a;
      cudaEvent_t event_sort_b;
      HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_sort_a, cudaEventDisableTiming));
      HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&event_sort_b, cudaEventDisableTiming));
      HANDLE_CUDA_ERROR(cudaEventRecord(event_sort_a, stream_sort_a));
      HANDLE_CUDA_ERROR(cudaEventRecord(event_sort_b, stream_sort_b));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream, event_sort_a));
      HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream, event_sort_b));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_sort_a));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_sort_b));
    }

    HANDLE_CUDA_ERROR(free(sort_offsets_temp_storage, stream));
    HANDLE_CUDA_ERROR(free(sort_indexes_temp_storage, stream));
    HANDLE_CUDA_ERROR(free(bucket_run_lengths, stream));
    HANDLE_CUDA_ERROR(free(bucket_run_offsets, stream));
    HANDLE_CUDA_ERROR(free(unique_bucket_indexes, stream));

    // aggregate buckets
    if (!dry_run) {
      if (copy_bases)
        HANDLE_CUDA_ERROR(cudaStreamWaitEvent(stream, event_bases_loaded));
      HANDLE_CUDA_ERROR(aggregate_buckets(is_first_loop, base_indexes, sorted_bucket_run_offsets, sorted_bucket_run_lengths, sorted_unique_bucket_indexes,
                                          copy_bases ? inputs_bases : ec.bases + inputs_offset, buckets_pass_one, buckets_count_pass_one, stream));
      if (copy_bases)
        HANDLE_CUDA_ERROR(cudaEventRecord(event_bases_free, stream));
    }
    HANDLE_CUDA_ERROR(free(base_indexes, stream));
    HANDLE_CUDA_ERROR(free(sorted_bucket_run_offsets, stream));
    HANDLE_CUDA_ERROR(free(sorted_bucket_run_lengths, stream));
    HANDLE_CUDA_ERROR(free(sorted_unique_bucket_indexes, stream));
    if (is_last_loop && copy_bases)
      HANDLE_CUDA_ERROR(free(inputs_bases, stream));
    inputs_offset += inputs_count;
    if (!is_first_loop && log_inputs_count < log_max_inputs_count)
      log_inputs_count++;
  }

  buckets top_buckets;
  HANDLE_CUDA_ERROR(allocate(top_buckets, windows_count_pass_one, pool, stream));

  if (!dry_run) {
    if (copy_scalars) {
      HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_copy_scalars));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_scalars_loaded));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_scalars_free));
    }
    if (copy_bases) {
      HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_copy_bases));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_bases_loaded));
      HANDLE_CUDA_ERROR(cudaEventDestroy(event_bases_free));
    }
    if (copy_scalars || copy_bases)
      HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_copy_finished));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_sort_a));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream_sort_b));
    if (top_window_unused_bits != 0) {
      const unsigned top_window_offset = (windows_count_pass_one - 1) << signed_bits_count_pass_one;
      const unsigned top_window_used_bits = signed_bits_count_pass_one - top_window_unused_bits;
      const unsigned top_window_used_buckets_count = 1 << top_window_used_bits;
      const unsigned top_window_unused_buckets_count = (1 << signed_bits_count_pass_one) - top_window_used_buckets_count;
      const unsigned top_window_unused_buckets_offset = top_window_offset + top_window_used_buckets_count;
      for (unsigned i = 0; i < top_window_unused_bits; i++)
        HANDLE_CUDA_ERROR(reduce_buckets(buckets_pass_one + top_window_offset, 1 << (signed_bits_count_pass_one - i - 1), stream));
      HANDLE_CUDA_ERROR(initialize_buckets(buckets_pass_one + top_window_unused_buckets_offset, top_window_unused_buckets_count, stream));
    }
    HANDLE_CUDA_ERROR(extract_top_buckets(buckets_pass_one, top_buckets, signed_bits_count_pass_one, windows_count_pass_one, stream));
  }

  unsigned source_bits_count = signed_bits_count_pass_one;
  unsigned source_windows_count = windows_count_pass_one;
  buckets source_buckets = buckets_pass_one;
  buckets_pass_one.ptr = nullptr;
  buckets target_buckets;
  for (unsigned i = 0;; i++) {
    const unsigned target_bits_count = (source_bits_count + 1) >> 1;
    const unsigned target_windows_count = source_windows_count << 1;
    const unsigned target_buckets_count = target_windows_count << target_bits_count;
    const unsigned log_data_split =
        get_optimal_log_data_split(cfg.device_props.multiProcessorCount, source_bits_count, target_bits_count, target_windows_count);
    const unsigned total_buckets_count = target_buckets_count << log_data_split;
    HANDLE_CUDA_ERROR(allocate(target_buckets, total_buckets_count, pool, stream));
    if (!dry_run)
      HANDLE_CUDA_ERROR(split_windows(source_bits_count, source_windows_count, source_buckets, target_buckets, total_buckets_count, stream));
    HANDLE_CUDA_ERROR(free(source_buckets, stream));

    if (!dry_run)
      for (unsigned j = 0; j < log_data_split; j++)
        HANDLE_CUDA_ERROR(reduce_buckets(target_buckets, total_buckets_count >> (j + 1), stream));
    if (target_bits_count == 1) {
      buckets_jacobian results;
      const unsigned result_windows_count = fd_q::MBC;
      if (copy_results)
        HANDLE_CUDA_ERROR(allocate(results, result_windows_count, pool, stream));
      if (!dry_run) {
        HANDLE_CUDA_ERROR(
            last_pass_gather(bits_count_pass_one, target_buckets, top_buckets, copy_results ? results : ec.results, result_windows_count, stream));
        if (copy_results) {
          HANDLE_CUDA_ERROR(cudaMemcpyAsync(ec.results, results, sizeof(point_jacobian) * result_windows_count, cudaMemcpyDeviceToHost, stream));
          if (ec.d2h_copy_finished)
            HANDLE_CUDA_ERROR(cudaEventRecord(ec.d2h_copy_finished, stream));
          if (ec.d2h_copy_finished_callback)
            HANDLE_CUDA_ERROR(cudaLaunchHostFunc(stream, ec.d2h_copy_finished_callback, ec.d2h_copy_finished_callback_data));
        }
      }
      if (copy_results)
        HANDLE_CUDA_ERROR(free(results, stream));
      HANDLE_CUDA_ERROR(free(target_buckets, stream));
      HANDLE_CUDA_ERROR(free(top_buckets, stream));

      break;
    }
    source_buckets = target_buckets;
    target_buckets.ptr = nullptr;
    source_bits_count = target_bits_count;
    source_windows_count = target_windows_count;
  }

  return cudaSuccess;
}

cudaError_t execute_async(const execution_configuration &exec_cfg) {
  int device_id;
  HANDLE_CUDA_ERROR(cudaGetDevice(&device_id));
  cudaDeviceProp props{};
  HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&props, device_id));
  const unsigned log_scalars_count = exec_cfg.log_scalars_count;
  cudaPointerAttributes scalars_attributes{};
  cudaPointerAttributes bases_attributes{};
  cudaPointerAttributes results_attributes{};
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&scalars_attributes, exec_cfg.scalars));
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&bases_attributes, exec_cfg.bases));
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&results_attributes, exec_cfg.results));
  bool copy_scalars = scalars_attributes.type == cudaMemoryTypeUnregistered || scalars_attributes.type == cudaMemoryTypeHost;
  unsigned log_min_inputs_count = exec_cfg.force_min_chunk_size ? exec_cfg.log_min_chunk_size
                                  : copy_scalars                ? get_log_min_inputs_count(log_scalars_count)
                                                                : log_scalars_count;
  if (exec_cfg.force_max_chunk_size) {
    extended_configuration cfg = {
        exec_cfg, scalars_attributes, bases_attributes, results_attributes, min(log_min_inputs_count, exec_cfg.log_max_chunk_size), exec_cfg.log_max_chunk_size,
        props};
    return schedule_execution(cfg, false);
  }
  unsigned log_max_inputs_count = copy_scalars ? max(log_min_inputs_count, log_scalars_count == 0 ? 0 : log_scalars_count - 1) : log_scalars_count;
  while (true) {
    extended_configuration cfg = {exec_cfg, scalars_attributes, bases_attributes, results_attributes, log_min_inputs_count, log_max_inputs_count, props};
    cudaError_t error = schedule_execution(cfg, true);
    if (error == cudaErrorMemoryAllocation) {
      log_max_inputs_count--;
      if (!copy_scalars)
        log_min_inputs_count--;
      if (log_max_inputs_count < get_log_min_inputs_count(log_scalars_count))
        return cudaErrorMemoryAllocation;
      continue;
    }
    if (error != cudaSuccess)
      return error;
    return schedule_execution(cfg, false);
  }
}

cudaError_t tear_down() { return cudaSuccess; }

} // namespace msm

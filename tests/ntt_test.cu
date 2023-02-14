#include "allocator.cuh"
#include "bc.cuh"
#include "bellman-cuda.h"
#include "common.cuh"
#include "ff.cuh"
#include "ntt.cuh"
#include <algorithm>
#include <curand.h>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace allocator;
using namespace ntt;
using namespace std;
typedef fd_q fq;

class ntt_test : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_CUDA_SUCCESS(cudaDeviceReset()); }

  void TearDown() override { cudaDeviceReset(); }

  void set_up(const unsigned log_max_values_count) {
    ASSERT_CUDA_SUCCESS(ff::set_up(25, 14));
    ASSERT_CUDA_SUCCESS(ntt::set_up());
    const unsigned max_values_count = 1 << log_max_values_count;
    u_values = new fq::storage[max_values_count];
    ASSERT_CUDA_SUCCESS(cudaMalloc(&d_values, sizeof(fq::storage) * max_values_count));
    ASSERT_CUDA_SUCCESS(cudaMallocHost(&h_values, sizeof(fq::storage) * max_values_count));
  }

  void tear_down() {
    delete[] u_values;
    ASSERT_CUDA_SUCCESS(cudaFree(d_values));
    ASSERT_CUDA_SUCCESS(cudaFreeHost(h_values));
    ASSERT_CUDA_SUCCESS(ntt::tear_down());
  }

  fq::storage *u_values{};
  fq::storage *d_values{};
  fq::storage *h_values{};

  void generate_values(const unsigned log_count) const {
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<fq>(d_values, 1 << log_count));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(h_values, d_values, sizeof(fq::storage) << log_count, cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(u_values, h_values, sizeof(fq::storage) << log_count, cudaMemcpyHostToHost));
  }

  void generate_values(const unsigned log_count, uint64_t seed) const {
    ASSERT_CUDA_SUCCESS(fields_populate_random_device<fq>(d_values, 1 << log_count, seed));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(h_values, d_values, sizeof(fq::storage) << log_count, cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(u_values, h_values, sizeof(fq::storage) << log_count, cudaMemcpyHostToHost));
  }

  static void log_data(fq::storage *m, const unsigned log_count, const string &filename, bool from_mont = false) {
    ofstream file;
    file.open(filename, ios::out);

    size_t n = 1 << log_count;

    file << "modulus (32-bit limbs expressed as uint32s):\n";
    fq::storage modulus{fq::CONFIG::modulus};
    for (unsigned limb : modulus.limbs)
      file << limb << "\n";

    file << "Data:\n";
    for (int i = 0; i < n; i++) {
      auto out = from_mont ? fq::from_montgomery(m[i]) : m[i];
      for (unsigned limb : out.limbs)
        // prefers \n over endl, so we don't flush the write buffer in the inner loop
        file << limb << "\n";
    }

    file.close();
  }

  static void verify_result(execution_configuration cfg) {
#define CHKSUM static constexpr fq::storage
    CHKSUM log_sz_10_nonbitrev_to_bitrev_fwd = {0x5521d69a, 0x426acab8, 0x7469620f, 0x01bc0865, 0x9a2d09af, 0x3eecf79d, 0x54213a5a, 0x078c59b1};
    CHKSUM log_sz_10_bitrev_to_nonbitrev_fwd = {0xcfb6a33b, 0x85267058, 0x17dcfc52, 0xad5e2d60, 0x8d7968b5, 0x8274dccd, 0x608f2da8, 0x0182d666};
    CHKSUM log_sz_10_nonbitrev_to_bitrev_inv = {0x3cd4cd57, 0x320b96d6, 0x221859ed, 0x3d3208bf, 0x00625015, 0xb898129d, 0x17fae9e1, 0x257ddfb2};
    CHKSUM log_sz_10_bitrev_to_nonbitrev_inv = {0x272fb4d8, 0xe1050c0f, 0x2490d8a2, 0x34df8912, 0x99e0cffa, 0x7dc2e7a7, 0x22f485af, 0x0e499759};
    CHKSUM log_sz_20_nonbitrev_to_bitrev_fwd = {0x93b945b1, 0x425df7b4, 0x573270ce, 0xe03b8d7a, 0x5a8815ff, 0xa2c6a7a9, 0x2dbfd4ec, 0x1656946a};
    CHKSUM log_sz_20_bitrev_to_nonbitrev_fwd = {0xe865ee6c, 0xfe45af67, 0x07c17782, 0xdc602b5d, 0xcbaf2c1e, 0x58f23786, 0x549f3348, 0x2e0d9a12};
    CHKSUM log_sz_20_nonbitrev_to_bitrev_inv = {0x7e1a012a, 0x567a9749, 0x1172b20d, 0x9a79fa66, 0xb4513ea3, 0x458b8662, 0xdb1a4774, 0x188ff72d};
    CHKSUM log_sz_20_bitrev_to_nonbitrev_inv = {0x7f42bc8f, 0x41325cfe, 0x416e4477, 0xaacfe5ee, 0x11c38b4b, 0x2438ba7a, 0xabfaf8eb, 0x0dd0ee55};
#undef CHKSUM

    fq::storage checksum{};
    switch (cfg.log_values_count) {
    case 10:
      checksum = cfg.bit_reversed_inputs ? (cfg.inverse ? log_sz_10_bitrev_to_nonbitrev_inv : log_sz_10_bitrev_to_nonbitrev_fwd)
                                         : (cfg.inverse ? log_sz_10_nonbitrev_to_bitrev_inv : log_sz_10_nonbitrev_to_bitrev_fwd);
      break;
    case 20:
      checksum = cfg.bit_reversed_inputs ? (cfg.inverse ? log_sz_20_bitrev_to_nonbitrev_inv : log_sz_20_bitrev_to_nonbitrev_fwd)
                                         : (cfg.inverse ? log_sz_20_nonbitrev_to_bitrev_inv : log_sz_20_nonbitrev_to_bitrev_fwd);
      break;
    default:
      FAIL() << "cfg.log_values_count = " << cfg.log_values_count << ", but we only have checksums for 10 and 20.";
    }

    fq::storage sum{};
    for (unsigned i = 0; i < (1 << cfg.log_values_count); i++) {
      sum = fq::add(sum, fq::mul(i, cfg.outputs[i]));
    }

    ASSERT_PRED2(fq::eq, sum, checksum);
  }

  void correctness(const unsigned log_count, bool bitrev_to_nonbitrev, bool inv, bool write_data_to_file = false) {
    // write_data_to_file = true;
    set_up(log_count);
    generate_values(log_count, 42);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    execution_configuration cfg = {mem_pool, stream, h_values, h_values, log_count, bitrev_to_nonbitrev, inv};
    if (write_data_to_file) {
      stringstream ss;
      ss << "inputs-" << log_count << "-" << bitrev_to_nonbitrev << "-" << inv << ".txt";
      log_data(h_values, log_count, ss.str(), true);
    }
    ASSERT_CUDA_SUCCESS(execute_async(cfg));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    if (write_data_to_file) {
      stringstream ss;
      ss << "outputs-" << log_count << "-" << bitrev_to_nonbitrev << "-" << inv << ".txt";
      log_data(h_values, log_count, ss.str(), true);
    }
    verify_result(cfg);
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  static void correctness_lde(const unsigned log_values_count, const unsigned log_extension_degree, const fd_q::storage *inputs,
                              fd_q::storage *expected_outputs) {
    typedef fd_q::storage storage;
    typedef allocation<storage> values;
    ff::set_up(14, 14);
    ntt::set_up();
    auto h_outputs = new storage[1 << (log_values_count + log_extension_degree)];
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream{};
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    const size_t inputs_size = sizeof(storage) << log_values_count;
    values d_inputs;
    ASSERT_CUDA_SUCCESS(allocate(d_inputs, 1 << log_values_count, mem_pool, stream));
    ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(d_inputs, inputs, inputs_size, cudaMemcpyHostToDevice, stream));
    cudaEvent_t copy_finished{};
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&copy_finished));
    ASSERT_CUDA_SUCCESS(cudaEventRecord(copy_finished, stream));
    for (unsigned i = 0; i < (1 << log_extension_degree); i++) {
      cudaStream_t ntt_stream{};
      ASSERT_CUDA_SUCCESS(cudaStreamCreate(&ntt_stream));
      ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(ntt_stream, copy_finished));
      storage *outputs = h_outputs + (i << log_values_count);
      const unsigned coset_index = i;
      execution_configuration cfg{};
      cfg.mem_pool = mem_pool;
      cfg.stream = ntt_stream;
      cfg.inputs = d_inputs;
      cfg.outputs = outputs;
      cfg.log_values_count = log_values_count;
      cfg.log_extension_degree = log_extension_degree;
      cfg.coset_index = coset_index;
      ASSERT_CUDA_SUCCESS(execute_async(cfg));
      cudaEvent_t ntt_finished{};
      ASSERT_CUDA_SUCCESS(cudaEventCreate(&ntt_finished));
      ASSERT_CUDA_SUCCESS(cudaEventRecord(ntt_finished, ntt_stream));
      ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(stream, ntt_finished));
      ASSERT_CUDA_SUCCESS(cudaEventDestroy(ntt_finished));
      ASSERT_CUDA_SUCCESS(cudaStreamDestroy(ntt_stream));
    }
    ASSERT_CUDA_SUCCESS(free(d_inputs, stream));
    ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream));
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    ASSERT_CUDA_SUCCESS(ntt::tear_down());
    for (unsigned i = 0; i < (1 << log_extension_degree); i++)
      ASSERT_PRED2(fq::eq, h_outputs[i], expected_outputs[i]);
    delete[] h_outputs;
  }

  void benchmark(const vector<unsigned> &log_counts, const vector<cudaMemoryType> &types, const vector<bool> &bitrevs, const vector<bool> &inverses) {
    const unsigned max_log_count = *max_element(log_counts.begin(), log_counts.end());
    set_up(max_log_count);
    generate_values(max_log_count);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, d_values, d_values, max_log_count, false, false}));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    printf("size");
    for (cudaMemoryType type : types) {
      string type_label;
      switch (type) {
      case cudaMemoryTypeUnregistered:
        type_label = "pag";
        break;
      case cudaMemoryTypeHost:
        type_label = "pin";
        break;
      case cudaMemoryTypeDevice:
        type_label = "dev";
        break;
      default:
        FAIL();
      }
      for (bool bitrev : bitrevs) {
        string bitrev_label = bitrev ? "b2n" : "n2b";
        for (bool inverse : inverses) {
          string inverse_label = inverse ? "inv" : "fwd";
          printf("\t%s_%s_%s", type_label.c_str(), bitrev_label.c_str(), inverse_label.c_str());
        }
      }
    }
    printf("\n");
    for (unsigned log_count : log_counts) {
      printf("2^%d", log_count);
      for (cudaMemoryType type : types) {
        fq::storage *values;
        switch (type) {
        case cudaMemoryTypeUnregistered:
          values = u_values;
          break;
        case cudaMemoryTypeHost:
          values = h_values;
          break;
        case cudaMemoryTypeDevice:
          values = d_values;
          break;
        default:
          FAIL();
        }
        for (bool bitrev : bitrevs) {
          for (bool inverse : inverses) {
            execution_configuration cfg = {mem_pool, stream, values, values, log_count, bitrev, inverse};
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
        }
      }
      printf("\n");
    }
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  void correctness_forward_inverse(const unsigned log_count) {
    set_up(log_count);
    generate_values(log_count);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, h_values, d_values, log_count, false, false, true}));
    ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, d_values, h_values, log_count, true, true, true}));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    for (unsigned i = 0; i < (1 << log_count); i++)
      ASSERT_PRED2(fq::eq, u_values[i], h_values[i]);
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  void correctness_lde_forward_inverse(const unsigned log_count) {
    set_up(log_count);
    generate_values(log_count);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    for (unsigned coset_index = 0; coset_index < 4; coset_index++) {
      ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, h_values, d_values, log_count, false, false, true, 2, coset_index}));
      ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, d_values, h_values, log_count, true, true, true, 2, coset_index}));
      ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
      for (unsigned i = 0; i < (1 << log_count); i++)
        ASSERT_PRED2(fq::eq, u_values[i], h_values[i]);
    }
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  void benchmark_lde(const vector<unsigned> &log_counts, const vector<unsigned> &log_extensions, const vector<cudaMemoryType> &types) {
    typedef fd_q::storage storage;
    typedef allocation<storage> values;
    const unsigned max_log_count = *max_element(log_counts.begin(), log_counts.end());
    const unsigned max_log_extension = *max_element(log_extensions.begin(), log_extensions.end());
    unsigned max_log_size = min(max_log_count + max_log_extension, 28);
    set_up(max_log_size);
    generate_values(max_log_size);
    cudaMemPool_t mem_pool;
    ASSERT_CUDA_SUCCESS(bc::mem_pool_create(mem_pool, 0));
    cudaStream_t stream;
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream));
    ASSERT_CUDA_SUCCESS(execute_async({mem_pool, stream, d_values, d_values, max_log_size}));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    printf("%10s", "size");
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
      const unsigned count = 1 << log_count;
      for (unsigned log_extension : log_extensions) {
        if (log_count + log_extension > 28)
          continue;
        const unsigned extension = 1 << log_extension;
        printf("2^%2d->2^%2d", log_count, log_count + log_extension);
        for (cudaMemoryType type : types) {
          storage *x_values;
          switch (type) {
          case cudaMemoryTypeUnregistered:
            x_values = u_values;
            break;
          case cudaMemoryTypeHost:
            x_values = h_values;
            break;
          case cudaMemoryTypeDevice:
            x_values = d_values;
            break;
          default:
            FAIL();
          }
          values a_inputs;
          values a_outputs;
          storage *inputs;
          storage *outputs;
          ASSERT_CUDA_SUCCESS(allocate(a_inputs, count, mem_pool, stream));
          inputs = a_inputs;
          if (type == cudaMemoryTypeDevice) {
            ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(a_inputs, d_values, sizeof(storage) << log_count, cudaMemcpyDeviceToDevice, stream));
            outputs = d_values;
          }
          cudaEvent_t start;
          cudaEvent_t end;
          ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
          ASSERT_CUDA_SUCCESS(cudaEventCreate(&end));
          ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream));
          if (type != cudaMemoryTypeDevice) {
            ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(a_inputs, x_values, sizeof(storage) << log_count, cudaMemcpyHostToDevice, stream));
            ASSERT_CUDA_SUCCESS(allocate(a_outputs, count * extension, mem_pool, stream));
            outputs = a_outputs;
          }
          cudaEvent_t previous_step_finished;
          ASSERT_CUDA_SUCCESS(cudaEventCreate(&previous_step_finished));
          ASSERT_CUDA_SUCCESS(cudaEventRecord(previous_step_finished, stream));
          vector<cudaStream_t> streams;
          for (unsigned i = 0; i < extension; i++) {
            cudaStream_t ntt_stream;
            ASSERT_CUDA_SUCCESS(cudaStreamCreate(&ntt_stream));
            streams.push_back(ntt_stream);
            ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(ntt_stream, previous_step_finished));
            ASSERT_CUDA_SUCCESS(cudaEventDestroy(previous_step_finished));
            const unsigned offset = i << log_count;
            execution_configuration cfg = {mem_pool, ntt_stream, inputs, outputs + offset, log_count, false, false, false, log_extension, i};
            ASSERT_CUDA_SUCCESS(execute_async(cfg));
            if (i != extension - 1) {
              ASSERT_CUDA_SUCCESS(cudaEventCreate(&previous_step_finished));
              ASSERT_CUDA_SUCCESS(cudaEventRecord(previous_step_finished, ntt_stream));
            }
          }
          for (unsigned i = 0; i < extension; i++) {
            cudaStream_t ntt_stream = streams[i];
            if (type != cudaMemoryTypeDevice) {
              const unsigned offset = i << log_count;
              ASSERT_CUDA_SUCCESS(cudaMemcpyAsync(x_values + offset, outputs + offset, sizeof(storage) << log_count, cudaMemcpyDeviceToHost, ntt_stream));
            }
            cudaEvent_t ntt_finished;
            ASSERT_CUDA_SUCCESS(cudaEventCreate(&ntt_finished));
            ASSERT_CUDA_SUCCESS(cudaEventRecord(ntt_finished, ntt_stream));
            ASSERT_CUDA_SUCCESS(cudaStreamWaitEvent(stream, ntt_finished));
            ASSERT_CUDA_SUCCESS(cudaEventDestroy(ntt_finished));
            ASSERT_CUDA_SUCCESS(cudaStreamDestroy(ntt_stream));
          }
          ASSERT_CUDA_SUCCESS(cudaEventRecord(end, stream));
          ASSERT_CUDA_SUCCESS(cudaEventSynchronize(end));
          float elapsed;
          ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, end));
          printf("\t%8.3f ms", elapsed);
          ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
          ASSERT_CUDA_SUCCESS(cudaEventDestroy(end));
          ASSERT_CUDA_SUCCESS(free(a_inputs, stream));
          ASSERT_CUDA_SUCCESS(free(a_outputs, stream));
        }
        printf("\n");
      }
    }
    ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream));
    ASSERT_CUDA_SUCCESS(cudaMemPoolDestroy(mem_pool));
    tear_down();
  }

  // Tests the simplest multigpu ntt case, where half the data is on device 0 and half is on device 1.
  void correctness_multigpu_matches_singlegpu(const unsigned log_values_count) {
    int deviceCount;
    ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
      GTEST_SKIP_("\nFound 1 visible device but the test requires 2.\n");

    const unsigned log_n_devs = 1;
    const unsigned n_devs = 1 << log_n_devs;
    const unsigned values_count = 1 << log_values_count;
    const size_t values_per_dev = values_count >> log_n_devs;

    // Deliberately tests the top-level "bellman-cuda" API.
    int dev_ids[n_devs] = {0, 1}; // Note: Some index logic below relies on dev_ids[0] == 0 and dev_ids[1] == 1.
    bc_mem_pool pools[n_devs];
    bc_stream streams[n_devs];
    fq::storage *d_values_control;
    fq::storage *d_values_per_dev[n_devs];
    fq::storage *h_values_control;
    fq::storage *h_values_per_dev[n_devs];

    ASSERT_BC_SUCCESS(bc_malloc_host((void **)&h_values_control, sizeof(fq::storage) * values_count));

    for (const auto d : dev_ids) {
      ASSERT_BC_SUCCESS(bc_malloc_host((void **)&h_values_per_dev[d], sizeof(fq::storage) * values_per_dev));
      ASSERT_BC_SUCCESS(bc_set_device(d));
      ASSERT_BC_SUCCESS(bc_device_enable_peer_access(d ^ 1));
      ASSERT_BC_SUCCESS(bc_stream_create(&streams[d], false));
      ASSERT_BC_SUCCESS(bc_mem_pool_create(&pools[d], d));
      ASSERT_BC_SUCCESS(bc_mem_pool_enable_peer_access(pools[d], d ^ 1));
      ASSERT_BC_SUCCESS(ff_set_up(25, 14));
      ASSERT_BC_SUCCESS(ntt_set_up());
      if (d == 0) {
        ASSERT_BC_SUCCESS(bc_malloc((void **)&d_values_control, sizeof(fq::storage) * values_count));
      }
      ASSERT_BC_SUCCESS(bc_malloc((void **)&d_values_per_dev[d], sizeof(fq::storage) * values_per_dev));
    }

    const vector<bool> bitrevs = {false, true};
    const vector<bool> inverses = {false, true};
    for (auto bit_reversed_inputs : bitrevs) {
      for (auto inverse : inverses) {
        ntt_configuration cfgs[n_devs] = {0};
        for (const auto d : dev_ids) {
          ASSERT_BC_SUCCESS(bc_set_device(d));
          if (d == 0) {
            fields_populate_random_device<fq>(d_values_control, values_count); // includes cudaDeviceSynchronize()
          }
          ASSERT_BC_SUCCESS(bc_memcpy_async(d_values_per_dev[d], &d_values_control[d * values_per_dev], sizeof(fq::storage) * values_per_dev, streams[d]));
          cfgs[d] = {pools[d], streams[d], d_values_per_dev[d], d_values_per_dev[d], log_values_count, bit_reversed_inputs, inverse, false};
        }
        ASSERT_BC_SUCCESS(ntt_execute_async_multigpu(cfgs, dev_ids, log_n_devs));
        {
          ASSERT_BC_SUCCESS(bc_set_device(0));
          ntt_configuration cfg_control = {pools[0], streams[0], d_values_control, d_values_control, log_values_count, bit_reversed_inputs, inverse, false};
          ASSERT_BC_SUCCESS(ntt_execute_async(cfg_control));
        }
        for (const auto d : dev_ids) {
          ASSERT_BC_SUCCESS(bc_set_device(d));
          if (d == 0) {
            ASSERT_BC_SUCCESS(bc_memcpy_async(h_values_control, d_values_control, sizeof(fq::storage) * values_count, streams[d]));
          }
          ASSERT_BC_SUCCESS(bc_memcpy_async(h_values_per_dev[d], d_values_per_dev[d], sizeof(fq::storage) * values_per_dev, streams[d]));
          ASSERT_BC_SUCCESS(bc_stream_synchronize(streams[d]));
        }
        for (unsigned d = 0; d < n_devs; d++)
          for (unsigned i = 0; i < values_per_dev; i++)
            ASSERT_PRED2(fq::eq, h_values_control[i + d * values_per_dev], h_values_per_dev[d][i]);
      }
    }

    ASSERT_BC_SUCCESS(bc_free_host(h_values_control));

    for (const auto d : dev_ids) {
      ASSERT_BC_SUCCESS(bc_free_host(h_values_per_dev[d]));
      ASSERT_BC_SUCCESS(bc_set_device(d));
      ASSERT_BC_SUCCESS(bc_stream_destroy(streams[d]));
      ASSERT_BC_SUCCESS(bc_mem_pool_destroy(pools[d]));
      if (d == 0) {
        ASSERT_BC_SUCCESS(bc_free(d_values_control));
      }
      ASSERT_BC_SUCCESS(bc_free(d_values_per_dev[d]));
    }
  }

  // Tests "4n ntt" case needed by the prover, where the data is divided into 8 chunks with
  // chunks 0, 2, 4, 6 residing on device 0 and chunks 1, 3, 5, 7 residing on device 1.
  void correctness_multigpu_matches_singlegpu_4n(const unsigned log_values_count) {
    int deviceCount;
    ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2)
      GTEST_SKIP_("\nFound 1 visible device but the test requires 2.\n");

    const unsigned log_n_devs = 3; // 8 "virtual" devices
    const unsigned n_devs = 1 << log_n_devs;
    const unsigned values_count = 1 << log_values_count;
    const size_t values_per_dev = values_count >> log_n_devs;

    // Deliberately tests the top-level "bellman-cuda" API.
    int cuda_dev_ids[2] = {0, 1};                         // Note: Some index logic below relies on dev_ids[0] == 0 and dev_ids[1] == 1.
    int chunk_dev_ids[n_devs] = {0, 1, 0, 1, 0, 1, 0, 1}; // Note: Some index logic below relies on dev_ids[0] == 0 and dev_ids[1] == 1.
    bc_mem_pool pools[2];
    bc_stream streams[n_devs];
    fq::storage *d_values_control;
    fq::storage *d_values_per_dev[n_devs];
    fq::storage *h_values_control;
    fq::storage *h_values_per_dev[n_devs];

    ASSERT_BC_SUCCESS(bc_malloc_host((void **)&h_values_control, sizeof(fq::storage) * values_count));

    for (const int d : cuda_dev_ids) {
      ASSERT_BC_SUCCESS(bc_set_device(d));
      ASSERT_BC_SUCCESS(bc_device_enable_peer_access(d ^ 1));
      ASSERT_BC_SUCCESS(bc_mem_pool_create(&pools[d], d));
      ASSERT_BC_SUCCESS(bc_mem_pool_enable_peer_access(pools[d], d ^ 1));
      ASSERT_BC_SUCCESS(ff_set_up(25, 14));
      ASSERT_BC_SUCCESS(ntt_set_up());
      if (d == 0) {
        ASSERT_BC_SUCCESS(bc_malloc((void **)&d_values_control, sizeof(fq::storage) * values_count));
      }
    }

    for (int d = 0; d < n_devs; d++) {
      const auto cuda_dev = chunk_dev_ids[d];
      ASSERT_BC_SUCCESS(bc_malloc_host((void **)&h_values_per_dev[d], sizeof(fq::storage) * values_per_dev));
      ASSERT_BC_SUCCESS(bc_set_device(cuda_dev));
      ASSERT_BC_SUCCESS(bc_stream_create(&streams[d], false));
      ASSERT_BC_SUCCESS(bc_malloc((void **)&d_values_per_dev[d], sizeof(fq::storage) * values_per_dev));
    }

    const vector<bool> bitrevs = {false, true};
    const vector<bool> inverses = {false, true};
    for (auto bit_reversed_inputs : bitrevs) {
      for (auto inverse : inverses) {
        ntt_configuration cfgs[n_devs] = {0};
        ASSERT_BC_SUCCESS(bc_set_device(0));
        fields_populate_random_device<fq>(d_values_control, values_count); // includes cudaDeviceSynchronize()
        for (int d = 0; d < n_devs; d++) {
          const auto cuda_dev = chunk_dev_ids[d];
          ASSERT_BC_SUCCESS(bc_set_device(cuda_dev));
          ASSERT_BC_SUCCESS(bc_memcpy_async(d_values_per_dev[d], &d_values_control[d * values_per_dev], sizeof(fq::storage) * values_per_dev, streams[d]));
          cfgs[d] = {pools[cuda_dev], streams[d], d_values_per_dev[d], d_values_per_dev[d], log_values_count, bit_reversed_inputs, inverse, false};
        }
        ASSERT_BC_SUCCESS(ntt_execute_async_multigpu(cfgs, chunk_dev_ids, log_n_devs));
        {
          ASSERT_BC_SUCCESS(bc_set_device(0));
          ntt_configuration cfg_control = {pools[0], streams[0], d_values_control, d_values_control, log_values_count, bit_reversed_inputs, inverse, false};
          ASSERT_BC_SUCCESS(ntt_execute_async(cfg_control));
          ASSERT_BC_SUCCESS(bc_memcpy_async(h_values_control, d_values_control, sizeof(fq::storage) * values_count, streams[0]));
        }
        for (int d = 0; d < n_devs; d++) {
          const auto cuda_dev = chunk_dev_ids[d];
          ASSERT_BC_SUCCESS(bc_set_device(cuda_dev));
          ASSERT_BC_SUCCESS(bc_memcpy_async(h_values_per_dev[d], d_values_per_dev[d], sizeof(fq::storage) * values_per_dev, streams[d]));
          ASSERT_BC_SUCCESS(bc_stream_synchronize(streams[d]));
        }
        for (unsigned d = 0; d < n_devs; d++)
          for (unsigned i = 0; i < values_per_dev; i++)
            ASSERT_PRED2(fq::eq, h_values_control[i + d * values_per_dev], h_values_per_dev[d][i]);
      }
    }

    ASSERT_BC_SUCCESS(bc_free_host(h_values_control));

    for (const auto d : cuda_dev_ids) {
      ASSERT_BC_SUCCESS(bc_set_device(d));
      ASSERT_BC_SUCCESS(bc_mem_pool_destroy(pools[d]));
      if (d == 0) {
        ASSERT_BC_SUCCESS(bc_free(d_values_control));
      }
    }

    for (int d = 0; d < n_devs; d++) {
      const auto cuda_dev = chunk_dev_ids[d];
      ASSERT_BC_SUCCESS(bc_free_host(h_values_per_dev[d]));
      ASSERT_BC_SUCCESS(bc_set_device(cuda_dev));
      ASSERT_BC_SUCCESS(bc_stream_destroy(streams[d]));
      ASSERT_BC_SUCCESS(bc_free(d_values_per_dev[d]));
    }
  }
};

TEST_F(ntt_test, correctness_forward_inverse_size_10) { correctness_forward_inverse(10); };

TEST_F(ntt_test, correctness_forward_inverse_size_20) { correctness_forward_inverse(20); };

TEST_F(ntt_test, correctness_lde_forward_inverse_size_10) { correctness_lde_forward_inverse(10); };

TEST_F(ntt_test, correctness_lde_forward_inverse_size_20) { correctness_lde_forward_inverse(20); };

TEST_F(ntt_test, correctness_size_10_nonbitrev_to_bitrev_forward) { correctness(10, false, false); }

TEST_F(ntt_test, correctness_size_10_bitrev_to_nonbitrev_forward) { correctness(10, true, false); }

TEST_F(ntt_test, correctness_size_10_nonbitrev_to_bitrev_inverse) { correctness(10, false, true); }

TEST_F(ntt_test, correctness_size_10_bitrev_to_nonbitrev_inverse) { correctness(10, true, true); }

TEST_F(ntt_test, correctness_size_20_nonbitrev_to_bitrev_forward) { correctness(20, false, false); }

TEST_F(ntt_test, correctness_size_20_bitrev_to_nonbitrev_forward) { correctness(20, true, false); }

TEST_F(ntt_test, correctness_size_20_nonbitrev_to_bitrev_inverse) { correctness(20, false, true); }

TEST_F(ntt_test, correctness_size_20_bitrev_to_nonbitrev_inverse) { correctness(20, true, true); }

TEST_F(ntt_test, correctness_lde_2) {
  // lde_factor: 2
  // log_degree: 2
  // coset factor Fr(0x0000000000000000000000000000000000000000000000000000000000000007)
  // domain generator for 2n: Fr(0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80)
  //
  // inputs
  // Fr(0x2a5685ec7047ab7815525dbe8fcfba949fa12d5a8e629a02967ceccb2bfded37)
  // Fr(0x2da239a4fa75e046b6048a40d7dc8a17d432d20d575883915707a0f0a1977cb9)
  // Fr(0x25bb97545bc1c461f371f64df42bcf1838a1afcf6ea11a43b4170adf5f1dfbca)
  // Fr(0x0e171dd55a940ef01c49266f586baff37a54b50d6fa65479a7f5bfa76842d4a7)
  //
  // expected
  // Fr(0x1ad69a5fa8dbb03b28cd4d40513fd753660f965fc476a6aa076318a28d6a7320)
  // Fr(0x1d7ac7369bf3a7a7c352c02f533d51166f7b33f11533bf94ab720cf7040bca55)
  // Fr(0x1d3afbcfc7f34d1111cc9ce6053f5c155b7156ecb309a3aa537992fac0468bf4)
  // Fr(0x23696bd8d32a68c29f0c86ee14010d762554abe4331ced900fc305046e3aeb72)
  // Fr(0x09127e80ce2f49ded0dc8b6c576ffe6b4509a50ff8aa9941d5579caeefe20f8b)
  // Fr(0x08393e1622238486ea4d3a4506960afaa93ebee64a8e313a3a61103f473d71b2)
  // Fr(0x12731193d74f3a39de1c0f575018cda930d005bee425f38587b2e26029b5e1d5)
  // Fr(0x24d2aca1371964ed4b6316848e1d62890f047b241eb8c8e63ac438b66f2251c8)
  const unsigned log_values_count = 2;
  const unsigned log_extension_degree = 1;
  typedef fd_q::storage storage;
  storage inputs[] = {{0x2bfded37, 0x967ceccb, 0x8e629a02, 0x9fa12d5a, 0x8fcfba94, 0x15525dbe, 0x7047ab78, 0x2a5685ec},
                      {0xa1977cb9, 0x5707a0f0, 0x57588391, 0xd432d20d, 0xd7dc8a17, 0xb6048a40, 0xfa75e046, 0x2da239a4},
                      {0x5f1dfbca, 0xb4170adf, 0x6ea11a43, 0x38a1afcf, 0xf42bcf18, 0xf371f64d, 0x5bc1c461, 0x25bb9754},
                      {0x6842d4a7, 0xa7f5bfa7, 0x6fa65479, 0x7a54b50d, 0x586baff3, 0x1c49266f, 0x5a940ef0, 0x0e171dd5}};
  storage expected_outputs[] = {{0x8d6a7320, 0x076318a2, 0xc476a6aa, 0x660f965f, 0x513fd753, 0x28cd4d40, 0xa8dbb03b, 0x1ad69a5f},
                                {0x040bca55, 0xab720cf7, 0x1533bf94, 0x6f7b33f1, 0x533d5116, 0xc352c02f, 0x9bf3a7a7, 0x1d7ac736},
                                {0xc0468bf4, 0x537992fa, 0xb309a3aa, 0x5b7156ec, 0x053f5c15, 0x11cc9ce6, 0xc7f34d11, 0x1d3afbcf},
                                {0x6e3aeb72, 0x0fc30504, 0x331ced90, 0x2554abe4, 0x14010d76, 0x9f0c86ee, 0xd32a68c2, 0x23696bd8},
                                {0xefe20f8b, 0xd5579cae, 0xf8aa9941, 0x4509a50f, 0x576ffe6b, 0xd0dc8b6c, 0xce2f49de, 0x09127e80},
                                {0x473d71b2, 0x3a61103f, 0x4a8e313a, 0xa93ebee6, 0x06960afa, 0xea4d3a45, 0x22238486, 0x08393e16},
                                {0x29b5e1d5, 0x87b2e260, 0xe425f385, 0x30d005be, 0x5018cda9, 0xde1c0f57, 0xd74f3a39, 0x12731193},
                                {0x6f2251c8, 0x3ac438b6, 0x1eb8c8e6, 0x0f047b24, 0x8e1d6289, 0x4b631684, 0x371964ed, 0x24d2aca1}};
  correctness_lde(log_values_count, log_extension_degree, inputs, expected_outputs);
}

TEST_F(ntt_test, correctness_lde_4) {
  // lde_factor=4
  // log_degree=2
  // coset factor Fr(0x0000000000000000000000000000000000000000000000000000000000000007)
  //
  // inputs
  // Fr(0x17f1cf514aa1af5be6ed823b41c083297fcccdc1343364fedbd46c982a2544f0)
  // Fr(0x167e6429f959654553aeb920fd3adae59bcd5379bf8550639935029de0c52ef5)
  // Fr(0x1c4572d990828d03da5307123475483b1483b8a83a23b8f7b8621ae73b7a12ac)
  // Fr(0x1b0a111cdf2dc3f5eb022fcd34ae3401f9b28d98bc3f81024b230238dc4ba07e)
  // domain generator: Fr(0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b)
  //
  // expected
  // Fr(0x0169d145bff5dbfa9449f8de7d09afe21070e68248825a199d3fbde8db3b2881)
  // Fr(0x0a55fe9f220934c30b29f536c40ef7f3add2c4facd8dbdcf4ec5c447ddca86fd)
  // Fr(0x091de7bae02e7d7b1545939cd497093f0de943015f0f8d274acf46a5c326898e)
  // Fr(0x1a85373287278f0d2eac41846fd103340ad2603de1f47e59f49af3f63c68dab3)
  // Fr(0x1ac896e77b9ce3cb221945ddce50ca922d6d9726f097ba413851133eeb0310b2)
  // Fr(0x07017ceee8e985f9d7f54c31f98291cf24184f85d3186ad6242edce994ad3c25)
  // Fr(0x1f755a4a5da596777a8947e978c457497e200ec2d2f0eba34f4176eeea14b99e)
  // Fr(0x1e87cf24685abd33271e2ef3c66a58fb2f8d41953a2c8340c3904b493ed00d4b)
  // Fr(0x02530d6ef28a2f259806aa152fa4ed17b63962c6af2cf67414dee4156c95e0b0)
  // Fr(0x007610f7a1475180ce367deb61de902c51f285c50ae21f7d68442d53169d468d)
  // Fr(0x1f7237ec6c2c16bad5a6595497277f0f0fdc16800a7ae7bde1cfe1fd2b26fbd4)
  // Fr(0x0d27987f495785e4a78241e15cd5b7f5bef74fb0928a25bacc7cc9670a3af0ae)
  // Fr(0x20a0882df1819c433740697064bc7333765019ba121a0bd540a90b1bd452f128)
  // Fr(0x251770011937f6549570fc9313aba11cff127aa0f7b9ecd791e25b3901693ffa)
  // Fr(0x17fbe492d7435a00297166154df974b95c70dd5edf4f6c11178d201d851d99ee)
  // Fr(0x021360834889d0d7a5933cd440a0839c2d5fc54ae7aa2f3d85392bee4dbb48b0)
  const unsigned log_values_count = 2;
  const unsigned log_extension_degree = 2;
  typedef fd_q::storage storage;
  storage inputs[] = {{0x2a2544f0, 0xdbd46c98, 0x343364fe, 0x7fcccdc1, 0x41c08329, 0xe6ed823b, 0x4aa1af5b, 0x17f1cf51},
                      {0xe0c52ef5, 0x9935029d, 0xbf855063, 0x9bcd5379, 0xfd3adae5, 0x53aeb920, 0xf9596545, 0x167e6429},
                      {0x3b7a12ac, 0xb8621ae7, 0x3a23b8f7, 0x1483b8a8, 0x3475483b, 0xda530712, 0x90828d03, 0x1c4572d9},
                      {0xdc4ba07e, 0x4b230238, 0xbc3f8102, 0xf9b28d98, 0x34ae3401, 0xeb022fcd, 0xdf2dc3f5, 0x1b0a111c}};
  storage expected_outputs[] = {{0xdb3b2881, 0x9d3fbde8, 0x48825a19, 0x1070e682, 0x7d09afe2, 0x9449f8de, 0xbff5dbfa, 0x0169d145},
                                {0xddca86fd, 0x4ec5c447, 0xcd8dbdcf, 0xadd2c4fa, 0xc40ef7f3, 0x0b29f536, 0x220934c3, 0x0a55fe9f},
                                {0xc326898e, 0x4acf46a5, 0x5f0f8d27, 0x0de94301, 0xd497093f, 0x1545939c, 0xe02e7d7b, 0x091de7ba},
                                {0x3c68dab3, 0xf49af3f6, 0xe1f47e59, 0x0ad2603d, 0x6fd10334, 0x2eac4184, 0x87278f0d, 0x1a853732},
                                {0xeb0310b2, 0x3851133e, 0xf097ba41, 0x2d6d9726, 0xce50ca92, 0x221945dd, 0x7b9ce3cb, 0x1ac896e7},
                                {0x94ad3c25, 0x242edce9, 0xd3186ad6, 0x24184f85, 0xf98291cf, 0xd7f54c31, 0xe8e985f9, 0x07017cee},
                                {0xea14b99e, 0x4f4176ee, 0xd2f0eba3, 0x7e200ec2, 0x78c45749, 0x7a8947e9, 0x5da59677, 0x1f755a4a},
                                {0x3ed00d4b, 0xc3904b49, 0x3a2c8340, 0x2f8d4195, 0xc66a58fb, 0x271e2ef3, 0x685abd33, 0x1e87cf24},
                                {0x6c95e0b0, 0x14dee415, 0xaf2cf674, 0xb63962c6, 0x2fa4ed17, 0x9806aa15, 0xf28a2f25, 0x02530d6e},
                                {0x169d468d, 0x68442d53, 0x0ae21f7d, 0x51f285c5, 0x61de902c, 0xce367deb, 0xa1475180, 0x007610f7},
                                {0x2b26fbd4, 0xe1cfe1fd, 0x0a7ae7bd, 0x0fdc1680, 0x97277f0f, 0xd5a65954, 0x6c2c16ba, 0x1f7237ec},
                                {0x0a3af0ae, 0xcc7cc967, 0x928a25ba, 0xbef74fb0, 0x5cd5b7f5, 0xa78241e1, 0x495785e4, 0x0d27987f},
                                {0xd452f128, 0x40a90b1b, 0x121a0bd5, 0x765019ba, 0x64bc7333, 0x37406970, 0xf1819c43, 0x20a0882d},
                                {0x01693ffa, 0x91e25b39, 0xf7b9ecd7, 0xff127aa0, 0x13aba11c, 0x9570fc93, 0x1937f654, 0x25177001},
                                {0x851d99ee, 0x178d201d, 0xdf4f6c11, 0x5c70dd5e, 0x4df974b9, 0x29716615, 0xd7435a00, 0x17fbe492},
                                {0x4dbb48b0, 0x85392bee, 0xe7aa2f3d, 0x2d5fc54a, 0x40a0839c, 0xa5933cd4, 0x4889d0d7, 0x02136083}};
  correctness_lde(log_values_count, log_extension_degree, inputs, expected_outputs);
}

TEST_F(ntt_test, benchmark_range) {
  const unsigned min_log_count = 21;
  const unsigned max_log_count = 28;
  vector<unsigned> log_counts(max_log_count - min_log_count + 1);
  iota(log_counts.begin(), log_counts.end(), min_log_count);
  const vector<cudaMemoryType> types = {cudaMemoryTypeDevice, cudaMemoryTypeHost, cudaMemoryTypeUnregistered};
  const vector<bool> bitrevs = {false, true};
  const vector<bool> inverses = {false, true};
  benchmark(log_counts, types, bitrevs, inverses);
}

TEST_F(ntt_test, benchmark_lde) {
  const unsigned min_log_count = 19;
  const unsigned max_log_count = 26;
  const unsigned min_log_extension = 1;
  const unsigned max_log_extension = 3;
  vector<unsigned> log_counts(max_log_count - min_log_count + 1);
  vector<unsigned> log_extensions(max_log_extension - min_log_extension + 1);
  iota(log_counts.begin(), log_counts.end(), min_log_count);
  iota(log_extensions.begin(), log_extensions.end(), min_log_extension);
  const vector<cudaMemoryType> types = {cudaMemoryTypeDevice, cudaMemoryTypeHost, cudaMemoryTypeUnregistered};
  benchmark_lde(log_counts, log_extensions, types);
}

TEST_F(ntt_test, correctness_multigpu_matches_singlegpu_size_20) { correctness_multigpu_matches_singlegpu(20); }

TEST_F(ntt_test, correctness_multigpu_matches_singlegpu_size_26) { correctness_multigpu_matches_singlegpu(26); }

TEST_F(ntt_test, correctness_multigpu_matches_singlegpu_4n_size_20) { correctness_multigpu_matches_singlegpu_4n(20); }

TEST_F(ntt_test, correctness_multigpu_matches_singlegpu_4n_size_26) { correctness_multigpu_matches_singlegpu_4n(26); }

TEST_F(ntt_test, benchmark_multigpu) {
  int deviceCount;
  ASSERT_CUDA_SUCCESS(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2)
    GTEST_SKIP_("\nFound 1 visible device but the test requires 2.\n");

  const unsigned log_n_devs = 1;
  const unsigned n_devs = 1 << log_n_devs;
  const unsigned log_values_count = 28;
  const unsigned values_count = 1 << log_values_count;
  const size_t values_per_dev = values_count >> log_n_devs;
  const unsigned reps = 5;

  // Deliberately tests the top-level "bellman-cuda" API.
  int dev_ids[n_devs] = {0, 1}; // Note: Some index logic below relies on dev_ids[0] == 0 and dev_ids[1] == 1.
  bc_mem_pool pools[n_devs];
  bc_stream streams[n_devs];
  bc_event events_start[n_devs];
  bc_event events_end[n_devs];
  fq::storage *d_values_per_dev[n_devs];

  for (const auto d : dev_ids) {
    ASSERT_BC_SUCCESS(bc_set_device(d));
    ASSERT_BC_SUCCESS(bc_device_enable_peer_access(d ^ 1));
    ASSERT_BC_SUCCESS(bc_stream_create(&streams[d], false));
    ASSERT_BC_SUCCESS(bc_event_create(&events_start[d], false, false));
    ASSERT_BC_SUCCESS(bc_event_create(&events_end[d], false, false));
    ASSERT_BC_SUCCESS(bc_mem_pool_create(&pools[d], d));
    ASSERT_BC_SUCCESS(bc_mem_pool_enable_peer_access(pools[d], d ^ 1));
    ASSERT_BC_SUCCESS(ff_set_up(25, 14));
    ASSERT_BC_SUCCESS(ntt_set_up());
    ASSERT_BC_SUCCESS(bc_malloc((void **)&d_values_per_dev[d], sizeof(fq::storage) * values_per_dev));
  }

  const vector<bool> bitrevs = {false, true};
  const vector<bool> inverses = {false, true};
  for (auto bit_reversed_inputs : bitrevs) {
    for (auto inverse : inverses) {
      ntt_configuration cfgs[n_devs] = {0};
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        fields_populate_random_device<fq>(d_values_per_dev[d], values_per_dev); // includes cudaDeviceSynchronize()
        cfgs[d] = {pools[d], streams[d], d_values_per_dev[d], d_values_per_dev[d], log_values_count, bit_reversed_inputs, inverse, false};
      }
      // Warmup. Helps ensure clocks are up to speed on systems where we can't lock clocks.
      for (int w = 0; w < reps; w++)
        ASSERT_BC_SUCCESS(ntt_execute_async_multigpu(cfgs, dev_ids, log_n_devs));
      // pushing cross-device sync after warmup work improves gpu timeline alignment across devices
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        ASSERT_BC_SUCCESS(bc_event_record(events_start[d], streams[d]));
      }
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        ASSERT_BC_SUCCESS(bc_stream_wait_event(streams[d], events_start[d ^ 1]));
      }
      // now record start events for real
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        ASSERT_BC_SUCCESS(bc_event_record(events_start[d], streams[d]));
      }
      for (int r = 0; r < reps; r++)
        ASSERT_BC_SUCCESS(ntt_execute_async_multigpu(cfgs, dev_ids, log_n_devs));
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        ASSERT_BC_SUCCESS(bc_event_record(events_end[d], streams[d]));
      }
      for (const auto d : dev_ids) {
        ASSERT_BC_SUCCESS(bc_set_device(d));
        ASSERT_BC_SUCCESS(bc_event_synchronize(events_end[d]));
      }
      float ms_elapsed[n_devs];
      for (const auto d : dev_ids)
        ASSERT_BC_SUCCESS(bc_event_elapsed_time(&ms_elapsed[d], events_start[d], events_end[d]));
      string bitrev_label = bit_reversed_inputs ? "b2n" : "n2b";
      string inverse_label = inverse ? "inv" : "fwd";
      std::cout << bitrev_label << " " << inverse_label << std::endl;
      for (const auto d : dev_ids)
        printf("dev %d took %8.3f ms\n", d, ms_elapsed[d] / reps);
    }
  }

  for (const auto d : dev_ids) {
    ASSERT_BC_SUCCESS(bc_set_device(d));
    ASSERT_BC_SUCCESS(bc_stream_destroy(streams[d]));
    ASSERT_BC_SUCCESS(bc_event_destroy(events_start[d]));
    ASSERT_BC_SUCCESS(bc_event_destroy(events_end[d]));
    ASSERT_BC_SUCCESS(bc_mem_pool_destroy(pools[d]));
    ASSERT_BC_SUCCESS(bc_free(d_values_per_dev[d]));
  }
}

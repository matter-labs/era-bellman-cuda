#include "bellman-cuda.h"
#include "common.cuh"
#include "msm_bases.cuh"

class bc_test : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_CUDA_SUCCESS(cudaDeviceReset()); }

  void TearDown() override { cudaDeviceReset(); }

  // one-time setup
  void set_up() {
    ASSERT_BC_SUCCESS(bc_mem_pool_create(&pool, 0));                                 // create a memory pool for the use by the library routines
    msm::point_affine h_bases[256];                                                  // get the array of bases for msm on the host
    for (unsigned i = 0; i < 256; i++)                                               //
      h_bases[i] = msm::point_affine::to_montgomery(g_bases[i], fd_p());             //
    ASSERT_BC_SUCCESS(bc_malloc(&d_bases, sizeof(msm::point_affine) * 256));         // allocate memory for bases on the device
    ASSERT_BC_SUCCESS(bc_memcpy(d_bases, h_bases, sizeof(msm::point_affine) * 256)); // copy bases from host to device
    ASSERT_BC_SUCCESS(ff_set_up(25, 14));                                            // set_up ff before first use
    ASSERT_BC_SUCCESS(ntt_set_up());                                                 // set_up ntt before first use
    ASSERT_BC_SUCCESS(msm_set_up());                                                 // set_up msm before first use
  }

  // cleanup before exit
  void tear_down() {
    ASSERT_BC_SUCCESS(bc_mem_pool_destroy(pool)); // destroy the memory pool
    ASSERT_BC_SUCCESS(bc_free(d_bases));          // release the memory for the bases on the device
    ASSERT_BC_SUCCESS(msm_tear_down());           // tear down msm after last use
    ASSERT_BC_SUCCESS(ntt_tear_down());           // tear down ntt after last use
    ASSERT_BC_SUCCESS(ff_tear_down());            // tear down ntt after last use
  }

  bc_mem_pool pool{};
  void *d_bases{};
};

TEST_F(bc_test, simple_msm) {
  set_up();

  fd_q::storage scalars[256];                                               // array of scalars on the host that will be the input to the msm
  msm::point_jacobian results[256];                                         // array of jacobian EC points that will receive the results of msm
  bc_stream stream{};                                                       //
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                       // create a stream with the blocking_sync flag
  msm_configuration msm_cfg = {pool, stream, d_bases, scalars, results, 8}; // set the parameters of the msm execution
  ASSERT_BC_SUCCESS(msm_execute_async(msm_cfg));                            // schedule the msm execution on the stream
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                         // wait for the stream to finish all scheduled work
                                                                            // at this point results contains the msm results
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                             // destroy the stream

  tear_down();
}

TEST_F(bc_test, simple_ntt) {
  set_up();

  fd_q::storage values[256];                                                   // array of field elements that will be an input into the ntt
                                                                               // and also receive the output of the ntt
  bc_stream stream{};                                                          //
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                          // create a stream with the blocking_sync flag
  ntt_configuration ntt_cfg = {pool, stream, values, values, 8, false, false}; // set the parameters of the ntt execution
  ASSERT_BC_SUCCESS(ntt_execute_async(ntt_cfg));                               // schedule the ntt execution on the stream
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                            // wait for the stream to finish all scheduled work
                                                                               // at this point values contains the ntt results
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                // destroy the stream

  tear_down();
}

TEST_F(bc_test, ntt_into_msm) {
  set_up();

  fd_q::storage h_values[256];                                                        // array of field elements on the host that will be used
                                                                                      // as an input into the ntt
  void *d_values;                                                                     // variable holding the pointer to the temporary array on the device
                                                                                      // that will be the input/output to/from the ntt
                                                                                      // and also the input into the msm
  msm::point_jacobian results[256];                                                   // array of jacobian EC points that will receive the results of the msm
  bc_stream stream{};                                                                 //
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                                 // create a stream with the blocking_sync flag
  const size_t values_size = sizeof(fd_q::storage) * 256;                             // size of h_values in bytes
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(&d_values, values_size, pool, stream)); // schedule allocation of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_values, h_values, values_size, stream));        // schedule a copy of h_values to the temporary d_values array
  ntt_configuration ntt_cfg = {pool, stream, d_values, d_values, 8, false, false};    // set the parameters of the ntt execution
  ASSERT_BC_SUCCESS(ntt_execute_async(ntt_cfg));                                      // schedule the ntt execution on the stream
  msm_configuration msm_cfg = {pool, stream, d_bases, d_values, results, 8};          // set the parameters of the msm execution
  ASSERT_BC_SUCCESS(msm_execute_async(msm_cfg));                                      // schedule the msm execution on the stream
  ASSERT_BC_SUCCESS(bc_free_async(d_values, stream));                                 // schedule the release of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                   // wait for the stream to finish all scheduled work
                                                                                      // at this point results contains the msm results
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                       // destroy the stream

  tear_down();
}

TEST_F(bc_test, ntt_into_four_msms_in_series) {
  set_up();

  fd_q::storage h_values[1024];                                                     // array of field elements on the host that will be used
                                                                                    // as an input into the ntt
  fd_q::storage *d_values;                                                          // variable holding the pointer to the temporary array on the device
                                                                                    // that will be the input/output to/from the ntt
                                                                                    // and also the input into the msms
  msm::point_jacobian results[4][256];                                              // array of arrays of jacobian EC points that will receive the results
                                                                                    // of each of the four msms
  bc_stream stream{};                                                               //
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                               // create a stream with the blocking_sync flag
  const size_t values_size = sizeof(fd_q::storage) * 1024;                          // size of h_values in bytes
  void **ptr = reinterpret_cast<void **>(&d_values);                                // cast address d_values to address to void pointer
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(ptr, values_size, pool, stream));     // schedule allocation of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_values, h_values, values_size, stream));      // schedule a copy of h_values to the temporary d_values array
  ntt_configuration ntt_cfg = {pool, stream, d_values, d_values, 10, false, false}; // set the parameters of the ntt execution
  ASSERT_BC_SUCCESS(ntt_execute_async(ntt_cfg));                                    // schedule the ntt execution on the stream
  for (unsigned i = 0; i < 4; i++) {                                                // repeat for each msm
    fd_q::storage *scalars = d_values + i * 256;                                    // compute address of the slice of scalars for this msm
    msm_configuration msm_cfg = {pool, stream, d_bases, scalars, results[i], 8};    // set the parameters of the msm execution
    ASSERT_BC_SUCCESS(msm_execute_async(msm_cfg));                                  // schedule the msm execution on the stream
  }                                                                                 //
  ASSERT_BC_SUCCESS(bc_free_async(d_values, stream));                               // schedule the release of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                 // wait for the stream to finish all scheduled work
                                                                                    // at this point results contains the msm results
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                     // destroy the stream

  tear_down();
}

TEST_F(bc_test, ntt_into_four_msms_in_parallel) {
  set_up();

  fd_q::storage h_values[1024];                                                      // array of field elements on the host that will be used
                                                                                     // as an input into the ntt
  fd_q::storage *d_values;                                                           // variable holding the pointer to the temporary array on the device
                                                                                     // that will be the input/output to/from the ntt
                                                                                     // and also the input into the msms
  msm::point_jacobian results[4][256];                                               // array of arrays of jacobian EC points that will receive the results
                                                                                     // of each of the four msms
  bc_stream stream{};                                                                // the main stream
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                                // create the above stream with the blocking_sync flag
  const size_t values_size = sizeof(fd_q::storage) * 1024;                           // size of h_values in bytes
  void **ptr = reinterpret_cast<void **>(&d_values);                                 // cast address d_values to address to void pointer
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(ptr, values_size, pool, stream));      // schedule allocation of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_values, h_values, values_size, stream));       // schedule a copy of h_values to the temporary d_values array
  ntt_configuration ntt_cfg = {pool, stream, d_values, d_values, 10, false, false};  // set the parameters of the ntt execution
  ASSERT_BC_SUCCESS(ntt_execute_async(ntt_cfg));                                     // schedule the ntt execution on the stream
  bc_event ntt_finished{};                                                           // event that will be triggered when the ntt execution has finished
  ASSERT_BC_SUCCESS(bc_event_create(&ntt_finished, true, true));                     // create the above event with blocking_sync
                                                                                     // and disable_timing flags set to true
  ASSERT_BC_SUCCESS(bc_event_record(ntt_finished, stream));                          // record the above event on the main stream
  for (unsigned i = 0; i < 4; i++) {                                                 // repeat for each msm
    bc_stream msm_stream{};                                                          // a stream on which the msm of this loop iteration will be executed
                                                                                     // having a separate stream for each msm allows them to run in parallel
    ASSERT_BC_SUCCESS(bc_stream_create(&msm_stream, true));                          // create the above stream
    ASSERT_BC_SUCCESS(bc_stream_wait_event(msm_stream, ntt_finished));               // make this stream wait until the ntt has finished
    fd_q::storage *scalars = d_values + i * 256;                                     // compute address of the slice of scalars for this msm
    msm_configuration msm_cfg = {pool, msm_stream, d_bases, scalars, results[i], 8}; // set the parameters of the msm execution
    ASSERT_BC_SUCCESS(msm_execute_async(msm_cfg));                                   // schedule the msm execution on the stream
    bc_event msm_finished{};                                                         // event that will be triggered after the msm execution has finished
    ASSERT_BC_SUCCESS(bc_event_create(&msm_finished, true, true));                   // crate the above event with blocking_sync
                                                                                     // and disable_timing flags set to true
    ASSERT_BC_SUCCESS(bc_event_record(msm_finished, msm_stream));                    // record the above event on the msm stream
    ASSERT_BC_SUCCESS(bc_stream_wait_event(stream, msm_finished));                   // make the main stream wait until this msm to be finished
    ASSERT_BC_SUCCESS(bc_event_destroy(msm_finished));                               // destroy the msm_finished event
    ASSERT_BC_SUCCESS(bc_stream_destroy(msm_stream));                                // destroy the msm_stream stream
  }                                                                                  //
  ASSERT_BC_SUCCESS(bc_free_async(d_values, stream));                                // schedule the release of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                  // wait for the stream to finish all scheduled work
                                                                                     // at this point results contains the results of all the msms
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                      // destroy the stream

  tear_down();
}

TEST_F(bc_test, ntt_lde) {
  typedef fd_q::storage storage;
  const unsigned log_values_count = 8;
  const unsigned log_extension_degree = 2;
  const unsigned values_count = 1 << log_values_count;
  const unsigned extension_degree = 1 << log_extension_degree;
  set_up();

  storage h_inputs[values_count];                                                      // array of field elements on the host that will be used
                                                                                       // as an input by the ntts
  storage h_outputs[values_count * extension_degree];                                  // array of field elements on the host that will be used
                                                                                       // as output by the ntts
  storage *d_inputs;                                                                   // variable holding the pointer to the temporary array on the device
                                                                                       // that will be the input for the ntts
  storage *d_outputs;                                                                  // variable holding the pointer to the temporary array on the device
                                                                                       // that will be the output for the ntts
  bc_stream stream{};                                                                  // the main stream
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                                  // create the above stream with the blocking_sync flag
  const size_t inputs_size = sizeof(storage) * values_count;                           // size of h_values in bytes
  const size_t outputs_size = sizeof(storage) * values_count * extension_degree;       // size of h_values in bytes
  void **p_inputs = reinterpret_cast<void **>(&d_inputs);                              // cast address d_inputs to address to void pointer
  void **p_outputs = reinterpret_cast<void **>(&d_outputs);                            // cast address d_outputs to address to void pointer
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(p_inputs, inputs_size, pool, stream));   // schedule allocation of the temporary d_inputs array
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(p_outputs, outputs_size, pool, stream)); // schedule allocation of the temporary d_outputs array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_inputs, h_inputs, inputs_size, stream));         // schedule a copy of h_values to the temporary d_inputs array
  bc_event previous_step_finished{};                                                   // event that will be triggered when the inputs have been copied
                                                                                       // from the host to the device or when previous ntt was computed
  ASSERT_BC_SUCCESS(bc_event_create(&previous_step_finished, true, true));             // create the above event with blocking_sync
  ASSERT_BC_SUCCESS(bc_event_record(previous_step_finished, stream));                  // record the above event on the main stream
  bc_stream ntt_streams[extension_degree];                                             // array holding the streams on which the ntts will be scheduled
  for (unsigned i = 0; i < extension_degree; i++) {                                    // repeat for each ntt
    bc_stream ntt_stream{};                                                            // a stream on which the ntt of this loop iteration will be executed
                                                                                       // having a separate stream for each ntt allows them to overlap execution
                                                                                       // with memory transfers
    ASSERT_BC_SUCCESS(bc_stream_create(&ntt_stream, true));                            // create the above stream
    ntt_streams[i] = ntt_stream;                                                       // store the above stream into the array
    ASSERT_BC_SUCCESS(bc_stream_wait_event(ntt_stream, previous_step_finished));       // make this stream wait until the host to device copy
                                                                                       // or previous ntt has finished
    ASSERT_BC_SUCCESS(bc_event_destroy(previous_step_finished));                       // destroy the previous event
    storage *outputs = d_outputs + i * values_count;                                   // compute address of the slice of outputs for this ntt
    ntt_configuration cfg{};                                                           // set the parameters of the ntt execution
    cfg.mem_pool = pool;                                                               //
    cfg.stream = ntt_stream;                                                           //
    cfg.inputs = d_inputs;                                                             //
    cfg.outputs = outputs;                                                             //
    cfg.log_values_count = log_values_count;                                           //
    cfg.log_extension_degree = log_extension_degree;                                   //
    cfg.coset_index = i;                                                               //
    ASSERT_BC_SUCCESS(ntt_execute_async(cfg));                                         // schedule the ntt execution on the stream
    if (i != extension_degree - 1) {                                                   // if this is not the last ntt
      ASSERT_BC_SUCCESS(bc_event_create(&previous_step_finished, true, true));         // create a new event to record the ntt execution in thins loop
      ASSERT_BC_SUCCESS(bc_event_record(previous_step_finished, ntt_stream));          // record the above event
    }                                                                                  //
  }                                                                                    //
  for (unsigned i = 0; i < extension_degree; i++) {                                    // repeat for each ntt
    bc_stream ntt_stream = ntt_streams[i];                                             // retrieve the stream for this loop
    unsigned offset = i * values_count;                                                // compute the output offset for this loop iteration
    storage *d_o = d_outputs + offset;                                                 // compute the output address in device memory
    storage *h_o = h_outputs + offset;                                                 // compute the output address in host memory
    ASSERT_BC_SUCCESS(bc_memcpy_async(h_o, d_o, inputs_size, ntt_stream));             // schedule the copy ouf outputs from device to host
    bc_event ntt_finished{};                                                           // event that will be triggered after the above d2h copy has finished
    ASSERT_BC_SUCCESS(bc_event_create(&ntt_finished, true, true));                     // crate the above event with blocking_sync
                                                                                       // and disable_timing flags set to true
    ASSERT_BC_SUCCESS(bc_event_record(ntt_finished, ntt_stream));                      // record the above event on the ntt stream
    ASSERT_BC_SUCCESS(bc_stream_wait_event(stream, ntt_finished));                     // make the main stream wait until this ntt to be finished
    ASSERT_BC_SUCCESS(bc_event_destroy(ntt_finished));                                 // destroy the ntt_finished event
    ASSERT_BC_SUCCESS(bc_stream_destroy(ntt_stream));                                  // destroy the ntt_stream stream
  }                                                                                    //
  ASSERT_BC_SUCCESS(bc_free_async(d_inputs, stream));                                  // schedule the release of the temporary d_inputs array
  ASSERT_BC_SUCCESS(bc_free_async(d_outputs, stream));                                 // schedule the release of the temporary d_outputs array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                    // wait for the stream to finish all scheduled work
                                                                                       // at this point results contains the results of all the ntts
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                        // destroy the stream

  tear_down();
}

TEST_F(bc_test, ff_grand_product) {
  set_up();
  const unsigned values_count = 256;
  typedef fd_q::storage storage;
  bc_stream stream{};                                                                    // the execution stream
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                                    // create the above stream with the blocking_sync flag
  const size_t values_size = sizeof(storage) * values_count;                             // size of h_values in bytes
  storage h_values[values_count];                                                        // array of field elements on the host that will be used
                                                                                         // as input and output of the grand product
  storage *d_values;                                                                     // variable holding the pointer to the temporary array on the device
  void **p_values = reinterpret_cast<void **>(&d_values);                                // cast address d_values to address to void pointer
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(p_values, values_size, pool, stream));     // schedule allocation of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_values, h_values, values_size, stream));           // schedule a copy of h_values to the temporary d_values array
  ff_grand_product_configuration cfg = {pool, stream, d_values, d_values, values_count}; // configure the grand product execution
  ASSERT_BC_SUCCESS(ff_grand_product(cfg));                                              // schedule the grand product execution
  ASSERT_BC_SUCCESS(bc_memcpy_async(h_values, d_values, values_size, stream));           // schedule a copy of the temporary d_values array into h_values
  ASSERT_BC_SUCCESS(bc_free_async(d_values, stream));                                    // schedule the release of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                      // wait for the stream to finish all scheduled work
                                                                                         // at this point h_values contains the grand product
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                          // destroy the stream
  tear_down();
}

TEST_F(bc_test, ff_inverse) {
  set_up();
  const unsigned values_count = 1024;
  typedef fd_q::storage storage;
  bc_stream stream{};                                                                // the execution stream
  ASSERT_BC_SUCCESS(bc_stream_create(&stream, true));                                // create the above stream with the blocking_sync flag
  const size_t values_size = sizeof(storage) * values_count;                         // size of h_values in bytes
  storage h_values[values_count];                                                    // array of field elements on the host that will be used
                                                                                     // as input and output of the batch inverse
  storage *d_values;                                                                 // variable holding the pointer to the temporary array on the device
  void **p_values = reinterpret_cast<void **>(&d_values);                            // cast address d_values to address to void pointer
  ASSERT_BC_SUCCESS(bc_malloc_from_pool_async(p_values, values_size, pool, stream)); // schedule allocation of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_memcpy_async(d_values, h_values, values_size, stream));       // schedule a copy of h_values to the temporary d_values array
  ff_inverse_configuration cfg = {pool, stream, d_values, d_values, values_count};   // configure the inverse execution
  ASSERT_BC_SUCCESS(ff_inverse(cfg));                                                // schedule the inverse execution
  ASSERT_BC_SUCCESS(bc_memcpy_async(h_values, d_values, values_size, stream));       // schedule a copy of the temporary d_values array into h_values
  ASSERT_BC_SUCCESS(bc_free_async(d_values, stream));                                // schedule the release of the temporary d_values array
  ASSERT_BC_SUCCESS(bc_stream_synchronize(stream));                                  // wait for the stream to finish all scheduled work
                                                                                     // at this point h_values contains the inverses
  ASSERT_BC_SUCCESS(bc_stream_destroy(stream));                                      // destroy the stream
  tear_down();
}

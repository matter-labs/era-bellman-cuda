#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// bellman-cuda API error types
typedef enum bc_error {
  bc_success = 0,                 // The API call returned with no errors. In the case of query calls,
                                  // this also means that the operation being queried is complete (see bc_event_query() and bc_stream_query()).
  bc_error_invalid_value = 1,     // This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
  bc_error_memory_allocation = 2, // The API call failed because it was unable to allocate enough memory to perform the requested operation.
  bc_error_not_ready = 600        // This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error,
                                  // but must be indicated differently than bc_success (which indicates completion). Calls that may return this value include
                                  // bc_event_query() and bc_stream_query().
} bc_error;

// bellman-cuda stream
typedef struct bc_stream {
  void *handle;
} bc_stream;

// bellman-cuda event
typedef struct bc_event {
  void *handle;
} bc_event;

// bellman-cuda memory pool
typedef struct bc_mem_pool {
  void *handle;
} bc_mem_pool;

// bellman-cuda host function
// user_data - Argument value passed to the function
typedef void (*bc_host_fn)(void *user_data);

// Returns the number of compute-capable devices.
// count - Returns the number of devices with compute capability greater or equal to 2.0
bc_error bc_get_device_count(int *count);

// Returns which device is currently being used.
// device_id - Returns the device on which the active host thread executes the device code.
bc_error bc_get_device(int *device_id);

// Set device to be used for GPU executions.
// device_id - Device on which the active host thread should execute the device code.
bc_error bc_set_device(int device_id);

// Create an asynchronous stream.
// stream - Pointer to new stream identifier
// blocking_sync - If true, the stream uses a synchronization primitive for synchronization instead of spinning.
bc_error bc_stream_create(bc_stream *stream, bool blocking_sync);

// Make a compute stream wait on an event.
// stream - Stream to wait
// event - Event to wait on
bc_error bc_stream_wait_event(bc_stream stream, bc_event event);

// Waits for stream tasks to complete.
// stream - Stream identifier
bc_error bc_stream_synchronize(bc_stream stream);

// Queries an asynchronous stream for completion status.
// stream - Stream identifier
// Returns bc_success if all operations in stream have completed, or bc_error_not_ready if not.
bc_error bc_stream_query(bc_stream stream);

// Destroys and cleans up an asynchronous stream.
// stream - Stream identifier
bc_error bc_stream_destroy(bc_stream stream);

// Enqueues a host function call in a stream.
// stream - Stream on which the host function call will be scheduled
// fn - The function to call once preceding stream operations are complete
// user_data - User-specified data to be passed to the function
bc_error bc_launch_host_fn(bc_stream stream, bc_host_fn fn, void *user_data);

// Creates an event object.
// event - Pointer to newly created event
// blocking_sync - If true, the event uses a synchronization primitive for synchronization instead of spinning.
// disable_timing - If true, timing is disabled for better performance.
bc_error bc_event_create(bc_event *event, bool blocking_sync, bool disable_timing);

// Records an event.
// event - Event to record
// stream - the stream in which to record the event
bc_error bc_event_record(bc_event event, bc_stream stream);

// Waits for an event to complete.
//   event - Event to wait for
bc_error bc_event_synchronize(bc_event event);

// Queries an event's status.
// event - Event to query
// Returns bc_success if all captured work has been completed, or bc_error_not_ready if any captured work is incomplete.
bc_error bc_event_query(bc_event event);

// Destroys an event object.
// event - Event to destroy
bc_error bc_event_destroy(bc_event event);

// Computes the elapsed time between events.
// ms - Time between start and end in ms
// start - Starting event
// end - Ending event
bc_error bc_event_elapsed_time(float *ms, bc_event start, bc_event end);

// Gets free and total device memory.
// free - Returned free memory in bytes
// total - Returned total memory in bytes
bc_error bc_mem_get_info(size_t *free, size_t *total);

// Allocate memory on the device.
// ptr- Pointer to allocated device memory
// size - Requested allocation size in bytes
bc_error bc_malloc(void **ptr, size_t size);

// Allocates page-locked memory on the host.
// ptr - Pointer to allocated host memory
// size - Requested allocation size in bytes
bc_error bc_malloc_host(void **ptr, size_t size);

// Frees memory on the device.
// ptr - Device pointer to memory to free
bc_error bc_free(void *ptr);

// Frees page-locked memory.
// ptr - Pointer to memory to free
bc_error bc_free_host(void *ptr);

// Registers an existing host memory range for use by CUDA.
// ptr - Host pointer to memory to page-lock
// size - Size in bytes of the address range to page-lock in bytes
bc_error bc_host_register(void *ptr, size_t size);

// Unregisters a memory range that was registered with cudaHostRegister.
// ptr - Host pointer to memory to unregister
bc_error bc_host_unregister(void *ptr);

// Disables direct access to memory allocations on a peer device.
// device_id - Peer device to disable direct access to
bc_error bc_device_disable_peer_access(int device_id);

// Enables direct access to memory allocations on a peer device.
// device_id - Peer device to enable direct access to
bc_error bc_device_enable_peer_access(int device_id);

// Copies data between host and device.
// dst - Destination memory address
// src - Source memory address
// count - Size in bytes to copy
bc_error bc_memcpy(void *dst, const void *src, size_t count);

// Copies data between host and device.
// dst - Destination memory address
// src - Source memory address
// count - Size in bytes to copy
// stream - Stream on which this operation will be scheduled
bc_error bc_memcpy_async(void *dst, const void *src, size_t count, bc_stream stream);

// Initializes or sets device memory to a value.
// ptr - Pointer to device memory
// value - Value to set for each byte of specified memory
// count - Size in bytes to set
bc_error bc_memset(void *ptr, int value, size_t count);

// Initializes or sets device memory to a value.
// ptr - Pointer to device memory
// value - Value to set for each byte of specified memory
// count - Size in bytes to set
// stream - Stream on which this operation will be scheduled
bc_error bc_memset_async(void *ptr, int value, size_t count, bc_stream stream);

// Creates a memory pool.
// pool - Pointer to newly created memory pool
// device_id - Device on which the allocations should reside
bc_error bc_mem_pool_create(bc_mem_pool *pool, int device_id);

// Destroys the specified memory pool.
// pool - The memory pool to destroy
bc_error bc_mem_pool_destroy(bc_mem_pool pool);

// Disables direct access to memory allocations in a memory pool.
// pool - Memory pool to disable direct access to
// device_id - Peer device to disable direct access from
bc_error bc_mem_pool_disable_peer_access(bc_mem_pool pool, int device_id);

// Enables direct access to memory allocations in a memory pool.
// pool - Memory pool to enable direct access to
// device_id - Peer device to enable direct access from
bc_error bc_mem_pool_enable_peer_access(bc_mem_pool pool, int device_id);

// Allocates memory from a specified pool with stream ordered semantics.
// ptr - Returned device pointer
// size - Requested allocation size in bytes
// memPool - The pool to allocate from
// stream - The stream establishing the stream ordering semantic
bc_error bc_malloc_from_pool_async(void **ptr, size_t size, bc_mem_pool pool, bc_stream stream);

// Frees memory with stream ordered semantics.
// ptr - Device pointer to memory to free
// stream - The stream establishing the stream ordering promise
bc_error bc_free_async(void *ptr, bc_stream stream);

// Initializes the internal state for FF computations.
// Should be called once per lifetime of the process.
// powers_of_w_coarse_log_count - log2 of the number of precomputed powers of omega at a coarse level
// powers_of_g_coarse_log_count - log2 of the number of precomputed powers of omega generator at a coarse level
bc_error ff_set_up(unsigned powers_of_w_coarse_log_count, unsigned powers_of_g_coarse_log_count);

// Sets the elements of the target vector to the specified value.
// target - Device memory pointer to the vector of elements that should be set to the specified value
// value - Pointer to the value that the target vector elements should be set to, can be host or device memory pointer
// count - Number of elements in the target vector
// stream - Stream on which this operation will be scheduled
bc_error ff_set_value(void *target, const void *value, unsigned count, bc_stream stream);

// Sets the elements of the target vector to zero.
// target - Device memory pointer to the vector of elements that should be set to zero
// count - Number of elements in the target vector
// stream - Stream on which this operation will be scheduled
bc_error ff_set_value_zero(void *target, unsigned count, bc_stream stream);

// Sets the elements of the target vector to one in montgomery form.
// target - device memory pointer to the vector of elements that should be set to one in montgomery form
// count - Number of elements in the target vector
// stream - Stream on which this operation will be scheduled
bc_error ff_set_value_one(void *target, unsigned count, bc_stream stream);

// Multiplies each element of the x vector by the scalar a and stores the result in the result vector.
// Point-wise result = a * x
// a - Pointer to the scalar by which each element of x will be multiplied, can be host or device memory pointer
// x- Device memory pointer to the vector of values that should be multiplied by the scalar a
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_ax(const void *a, const void *x, void *result, unsigned count, bc_stream stream);

// Adds a scalar to each element of the x vector and stores the result in the result vector.
// Point-wise result = a + x
// a - Pointer to the scalar that will be added to each element of x, can be host or device memory pointer
// x- Device memory pointer to the vector of values to which the scalar a should be added
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_a_plus_x(const void *a, const void *x, void *result, unsigned count, bc_stream stream);

// Performs point-wise addition between elements from x and y and stores the result in the result vector.
// Point-wise result = x + y
// x- Device memory pointer to the vector of values that should be added to y
// y- Device memory pointer to the vector of values that should be added to x
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_x_plus_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Performs point-wise addition between elements from x multiplied by the scalar a and y and stores the result in the result vector.
// point-wise result = a * x + y
// a - Pointer to the scalar by which each element of x will be multiplied, can be host or device memory pointer
// x - Device memory pointer to the vector of values that should be multiplied by the scalar a and added to y
// y - Device memory pointer to the vector of values that should be added to ax
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_ax_plus_y(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Performs point-wise subtraction between elements from x and y and stores the result in the result vector.
// Point-wise result = x - y
// x - Device memory pointer to the vector of values from which y should be subtracted
// y - Device memory pointer to the vector of values that should be subtracted from x
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_x_minus_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Performs point-wise subtraction between elements from x multiplied by the scalar a and y and stores the result in the result vector.
// point-wise result = a * x - y
// a - Pointer to the scalar by which each element of x will be multiplied, can be host or device memory pointer
// x - Device memory pointer to the vector of values that should be multiplied by the scalar a and from which y should be subtracted
// y - Device memory pointer to the vector of values that should be subtracted from ax
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_ax_minus_y(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Performs point-wise subtraction between elements from x and y multiplied by the scalar a and stores the result in the result vector.
// point-wise result = x - a * y
// a - Pointer to the scalar by which each element of y will be multiplied, can be host or device memory pointer
// x - Device memory pointer to the vector of values from which ay should be subtracted
// y - Device memory pointer to the vector of values that should be multiplied by the scalar a and which should be subtracted from x
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_x_minus_ay(const void *a, const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Performs point-wise multiplication between elements from x and y and stores the result in the result vector.
// Point-wise result = x * y
// x- Device memory pointer to the vector of values that should be multiplied by y
// y- Device memory pointer to the vector of values that should be multiplied by x
// result - Device memory pointer to the vector where the result of this operation will be stored
// count - Number of elements in the x, y and result vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_x_mul_y(const void *x, const void *y, void *result, unsigned count, bc_stream stream);

// Configuration for the grand product execution
typedef struct ff_grand_product_configuration {
  bc_mem_pool mem_pool; // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;     // The stream on which the execution will be scheduled
  void *inputs;         // Device memory pointer to the vector of inputs for this calculation
  void *outputs;        // Device memory pointer to the vector of outputs for this calculation
  unsigned count;       // Number of elements in the input and output vectors
} ff_grand_product_configuration;

// Schedules the grand product execution
bc_error ff_grand_product(ff_grand_product_configuration configuration);

// Configuration for the multiply by powers execution
typedef struct ff_multiply_by_powers_configuration {
  bc_mem_pool mem_pool; // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;     // The stream on which the execution will be scheduled
  void *inputs;         // Device memory pointer to the vector of inputs for this calculation
  void *base;           // Pointer to the single values that is the base of the powers, can be host or device memory pointer
  void *outputs;        // Device memory pointer to the vector of outputs for this calculation
  unsigned count;       // Number of elements in the input and output vectors
} ff_multiply_by_powers_configuration;

// Schedules the multiply by powers execution
bc_error ff_multiply_by_powers(ff_multiply_by_powers_configuration configuration);

// Configuration for the inverse execution
typedef struct ff_inverse_configuration {
  bc_mem_pool mem_pool; // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;     // The stream on which the execution will be scheduled
  void *inputs;         // Device memory pointer to the vector of inputs for this calculation
  void *outputs;        // Device memory pointer to the vector of outputs for this calculation
  unsigned count;       // Number of elements in the input and output vectors
} ff_inverse_configuration;

// Schedules the inverse execution
bc_error ff_inverse(ff_inverse_configuration configuration);

// Configuration for the polynomial evaluation execution
typedef struct ff_poly_evaluate_configuration {
  bc_mem_pool mem_pool; // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;     // The stream on which the execution will be scheduled
  void *values;         // Device memory pointer to the vector of polynomial coefficients for this calculation
  void *point;          // Pointer to the single point over which the polynomial should be evaluated, can be host or device memory pointer
  void *result;         // Pointer to the single value containing the result of the evaluation, can be host or device memory pointer
  unsigned count;       // Number of elements in the coefficients vector
} ff_poly_evaluate_configuration;

// Configuration for the execution of the sort of unsigned 32-bit values
typedef struct ff_sort_u32_configuration {
  bc_mem_pool mem_pool; // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;     // The stream on which the execution will be scheduled
  void *values;         // Device memory pointer to the vector of unsigned 32-bit values to be sorted
  void *sorted_values;  // Device memory pointer to the vector of sorted values, must not overlap with above values vector
  unsigned count;       // Number of elements to be sorted
} ff_sort_u32_configuration;

// Schedules the polynomial evaluation execution
bc_error ff_poly_evaluate(ff_poly_evaluate_configuration configuration);

// Fills the vector with powers of omega.
// target - Pointer to the vector where the result will be written
// log_degree - Log2 of the degree, omega = 2^log_degree root of unity
// offset - Offset in the series of powers
// count - Number of elements in the target vector
// inverse - If true, fill the vector with inverses of the powers
// bit_reversed - If true, fill the vector with the powers at bit-reversed indexes
// stream - Stream on which this operation will be scheduled
bc_error ff_get_powers_of_w(void *target, unsigned log_degree, unsigned offset, unsigned count, bool inverse, bool bit_reversed, bc_stream stream);

// Fills the vector with powers of the generator.
// target - Pointer to the vector where the result will be written
// log_degree - Log2 of the degree
// offset - Offset in the series of powers
// count - Number of elements in the target vector
// inverse - If true, fill the vector with inverses of the powers
// bit_reversed - If true, fill the vector with the powers at bit-reversed indexes
// stream - Stream on which this operation will be scheduled
bc_error ff_get_powers_of_g(void *target, unsigned log_degree, unsigned offset, unsigned count, bool inverse, bool bit_reversed, bc_stream stream);

// Shifts the values of the vector by the specified omega shift.
// values - Pointer to the vector containing the values to be shifted
// result - Pointer to the vector containing the shifted values
// log_degree - Log2 of the degree, omega = 2^log_degree root of unity
// shift - The amount of shift to be applied
// offset - Offset in the series of powers
// count - Number of elements in the vectors
// inverse - Perform inverse shift
// stream - Stream on which this operation will be scheduled
bc_error ff_omega_shift(const void *values, void *result, unsigned log_degree, unsigned shift, unsigned offset, unsigned count, bool inverse, bc_stream stream);

// Shuffles the elements of a vector by bit-reversing the indexes
// values - Pointer to the vector containing the values to be shuffled
// result - Pointer to the vector containing the shuffled  values
// count - Log2 of the number of elements in the vectors
// stream - Stream on which this operation will be scheduled
bc_error ff_bit_reverse(const void *values, void *result, unsigned log_count, bc_stream stream);

// Shuffles the elements of a vector by bit-reversing the indexes using multiple GPUs
// values - Pointer to an array of vectors containing the values to be shuffled
// results - Pointer to an array of vectors containing the shuffled  values
// count - Log2 of the total number of elements in the vectors
// streams - Pointer to an array of streams on which this operation will be scheduled
// device_ids - pointer to an array of device ids of the devices on which this operation will be executed
// log_devices_count - Log2 of the count of devices on which this operation will be executed
bc_error ff_bit_reverse_multigpu(const void **values, void **results, unsigned log_count, const bc_stream *streams, const int *device_ids,
                                 unsigned log_devices_count);

// Selects values from the source vector based on the indexes and stores them in the destination vector
// destination[i]=source[indexes[i]]
// source - source vector from which values read
// destination - destination vector into which values written
// indexes - vector of source indexes
// count - number of elements
// stream - Stream on which this operation will be scheduled
bc_error ff_select(const void *source, void *destination, const unsigned *indexes, unsigned count, bc_stream stream);

// Sorts a vector of unsigned 32-bit values
// configuration - The configuration for the execution
bc_error ff_sort_u32(ff_sort_u32_configuration configuration);

// release all resources associated with the internal state for FF computations
bc_error ff_tear_down();

// Initializes the internal state for polynomial computations
// Should be called once per lifetime of the process
bc_error pn_set_up();

// Configuration for the Generate permutation polynomials execution
typedef struct generate_permutation_polynomials_configuration {
  bc_mem_pool mem_pool;    // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;        // The stream on which the execution will be scheduled
  unsigned *indexes;       // Device pointer to the matrix of the variable indexes in column-major order, columns representing the corresponding polynomials
  void *scalars;           // Device pointer to the vector of scalars that the twiddles will be multiplied with, length must be equal to the number of columns
                           // of the above matrix
  void *target;            // Device pointer to matrix containing the permutation polynomials in column-major order, columns representing the corresponding
                           // polynomials
  unsigned columns_count;  // Number of columns in the matrix
  unsigned log_rows_count; // Log2 of the number of rows in the matrix
} generate_permutation_polynomials_configuration;

// Schedule the Generate permutation polynomials execution
// configuration - The configuration for the execution
bc_error pn_generate_permutation_polynomials(generate_permutation_polynomials_configuration configuration);

// Set values of a vector of field elements based on bits of a bitfield
// values - device pointer to the vector of field elements where the results will be written
// packed_bits - device pointer to the bitfield packed as a vector of 32-bit words
// count - number of bits in the bitfield
// stream - Stream on which this operation will be scheduled
bc_error pn_set_values_from_packed_bits(void *values, const void *packet_bits, unsigned count, bc_stream stream);

// release all resources associated with the internal state for polynomial computations
bc_error pn_tear_down();

// Initializes the internal state for MSM computations
// Should be called once per lifetime of the process
bc_error msm_set_up();

// Configuration for the MSM execution
typedef struct msm_configuration {
  bc_mem_pool mem_pool;                  // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;                      // The stream on which the execution will be scheduled
  void *bases;                           // Device pointer to the bases that will be used for this execution
  void *scalars;                         // Pointer to the scalars used by this execution, can be either pinned or pageable host memory or device memory pointer
  void *results;                         // Pointer to an array of 254 EC points in jacobian coordinates corresponding to the 254 bits of the final MSM result,
                                         // can be either pinned or pageable host memory or device memory pointer
  unsigned log_scalars_count;            // Log2 of the number of scalars
  bc_event h2d_copy_finished;            // An optional event that should be recorded after the Host to Device memory copy  has completed
  bc_host_fn h2d_copy_finished_callback; // An optional callback that should be executed after the Host to Device memory copy has completed
  void *h2d_copy_finished_callback_data; // User-defined data for the above callback
  bc_event d2h_copy_finished;            // An optional event that should be recorded after the Device to Host memory copy has completed
  bc_host_fn d2h_copy_finished_callback; // An optional callback that should be executed after the Device to Host memory copy has completed
  void *d2h_copy_finished_callback_data; // User-defined data for the above callback
} msm_configuration;

// Schedule the MSM execution.
// configuration - The configuration for the execution
//
// The result array contain 254 values, one value for each bit of the final result value.
// Code example of the final reduction:
//    const unsigned result_bits_count = 254;
//    point sum = point::point_at_infinity();
//    for (int i = 0; i < result_bits_count; i++) {
//      int index = result_bits_count - i - 1;
//      point bucket = results[index];
//      sum = i == 0 ? bucket : ec::add(ec::dbl(sum), bucket);
//    }
//    return sum;
bc_error msm_execute_async(msm_configuration configuration);

// release all resources associated with the internal state for MSM computations
bc_error msm_tear_down();

// Initializes the internal state for NTT computations.
// Should be called once per lifetime of the process.
// Depends on ff_set_up being already called
bc_error ntt_set_up();

typedef struct ntt_configuration {
  bc_mem_pool mem_pool;                  // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;                      // The stream on which the execution will be scheduled
  void *inputs;                          // Pointer to the inputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  void *outputs;                         // Pointer to the outputs of this execution, can be either pinned or pageable host memory or device memory pointer.
  unsigned log_values_count;             // Log2 of the number of values
  bool bit_reversed_inputs;              // True if the inputs are in bit-reversed order and the output will be not bit-reversed, false otherwise
  bool inverse;                          // True if thi is an inverse NTT
  bool can_overwrite_inputs;             // If the inputs are on the device and the outputs are on the host, if this flag is set,
                                         // the inputs will be used for in-place calculation, otherwise and additional memory will be allocated on device
  unsigned log_extension_degree;         // The log2 of degree of extension for which a coset should be applied, if the value is zero, no coset is applied
  unsigned coset_index;                  // The index of the coset that should be applied
  bc_event h2d_copy_finished;            // An optional event that should be recorded after the Host to Device memory copy  has completed
  bc_host_fn h2d_copy_finished_callback; // An optional callback that should be executed after the Host to Device memory copy has completed
  void *h2d_copy_finished_callback_data; // User-defined data for the above callback
  bc_event d2h_copy_finished;            // An optional event that should be recorded after the Device to Host memory copy has completed
  bc_host_fn d2h_copy_finished_callback; // An optional callback that should be executed after the Device to Host memory copy has completed
  void *d2h_copy_finished_callback_data; // User-defined data for the above callback
} ntt_configuration;

// Schedule the NTT execution.
// configuration - The configuration for the execution
bc_error ntt_execute_async(ntt_configuration configuration);

// Schedule a single NTT whose input and output data are split evenly across several GPUs.
// User must supply an array of per-device configurations. The "log_values_count" in each configuration
// must be the same, and refer to the size of the entire NTT.
// configurations - array of configuration for the execution
// dev_ids - array of device ids participating in the execution
// log_n_devs - log2 of the number of devices participating in the execution
bc_error ntt_execute_async_multigpu(const ntt_configuration *configurations, const int *dev_ids, unsigned log_n_devs);

// release all resources associated with the internal state for NTT computations
bc_error ntt_tear_down();

#ifdef __cplusplus
} // extern "C"
#endif

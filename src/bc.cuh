#pragma once

namespace bc {

cudaError stream_create(cudaStream_t &stream, bool blocking_sync);

cudaError mem_pool_create(cudaMemPool_t &mem_pool, int device_id);

} // namespace bc
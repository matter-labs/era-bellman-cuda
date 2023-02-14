add_library(bellman-cuda STATIC
        allocator.cu
        allocator.cuh
        bc.cu
        bc.cuh
        bellman-cuda.cu
        bellman-cuda.h
        bellman-cuda-cub.cu
        bellman-cuda-cub.cuh
        carry_chain.cuh
        common.cuh
        ec.cuh
        ff.cu
        ff.cuh
        ff_config.cu
        ff_config.cuh
        ff_dispatch_st.cuh
        ff_kernels.cu
        ff_kernels.cuh
        host_math.cuh
        memory.cuh
        msm.cu
        msm.cuh
        msm_kernels.cu
        msm_kernels.cuh
        ntt.cu
        ntt.cuh
        ntt_kernels.cu
        ntt_kernels.cuh
        pn.cu
        pn.cuh
        pn_kernels.cu
        pn_kernels.cuh
        ptx.cuh
        utils.cu
        utils.cuh
        )
set_target_properties(bellman-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if (NOT BUILD_TESTS)
    set_target_properties(bellman-cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    install(TARGETS bellman-cuda DESTINATION .)
endif ()

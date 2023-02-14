find_package(CUDAToolkit)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
add_executable(tests
        bc_test.cu
        common.cu
        common.cuh
        ec_test.cu
        ec_test_kernels.cu
        ec_test_kernels.cuh
        ff_test.cu
        ff_test_kernels.cu
        ff_test_kernels.cuh
        montmul_test.cu
        msm_bases.cuh
        msm_test.cu
        ntt_test.cu
        pn_test.cu
        tests.cu
        )
target_include_directories(tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(tests bellman-cuda)
target_link_libraries(tests gtest gtest_main CUDA::curand)
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
gtest_discover_tests(tests)

cmake_minimum_required(VERSION 3.21)

project(uf_gpu CUDA CXX)

enable_testing()

set(CMAKE_CUDA_ARCHITECTURES 70 80 90)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

add_executable(bench_main bench_main.cu cluster2d_kernels.cu)
target_compile_features(bench_main PUBLIC cuda_std_17)
target_compile_options(bench_main PUBLIC -lineinfo)

add_executable(bench_main_cpu bench_main_cpu.cpp)
target_compile_features(bench_main_cpu PUBLIC cxx_std_17)
target_link_libraries(bench_main_cpu PUBLIC OpenMP::OpenMP_CXX)

add_executable(bench_main_debug bench_main_debug.cpp)
target_compile_features(bench_main_debug PUBLIC cxx_std_17)

add_executable(test_main test_main.cu cluster2d_kernels.cu)
target_compile_features(test_main PUBLIC cuda_std_17)
target_link_libraries(test_main PUBLIC GTest::gtest_main OpenMP::OpenMP_CXX)
target_compile_options(test_main PUBLIC -lineinfo)

include(GoogleTest)
gtest_discover_tests(test_main)

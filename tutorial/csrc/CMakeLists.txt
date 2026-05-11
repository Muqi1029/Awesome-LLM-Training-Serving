cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CUDA_SRC_TUTORIAL LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
message(STATUS "Using CUDAToolkit VERSION: ${CUDAToolkit_VERSION}")

add_subdirectory(cccl)

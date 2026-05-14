cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(ext_test LANGUAGES CXX CUDA)

message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_ROOT= ${CMAKE_ROOT}")
message(STATUS "CMAKE_CUDA_COMPILER= ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER= ${CMAKE_CXX_COMPILER}")

include(FetchContent)


find_package(Python COMPONENTS Interpreter REQUIRED)
message(STATUS "Python_EXECUTABLE= ${Python_EXECUTABLE}")
set(CMAKE_PREFIX_PATH "/root/.envs/dev/lib/python3.12/site-packages")
find_package(Torch REQUIRED)


find_package(CUDAToolkit REQUIRED)
message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
message(STATUS "Detected CUDAToolkit_VERSION=${CUDAToolkit_VERSION}")

if ("${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "13.0")
    message("CUDAToolkit_VERSION ${CUDAToolkit_VERSION} >= 13.0")
elseif ("${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDAToolkit_VERSION ${CUDAToolkit_VERSION} >= 12.8")
elseif ("${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDAToolkit_VERSION ${CUDAToolkit_VERSION} >= 12.4")
elseif ("${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDAToolkit_VERSION ${CUDAToolkit_VERSION} >= 12.1")
elseif ("${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDAToolkit_VERSION ${CUDAToolkit_VERSION} >= 11.8")
endif()

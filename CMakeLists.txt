cmake_minimum_required(VERSION 3.27)
project(
  NESO-Particles-Python
  VERSION 0.0.1
  LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

option(NESO_PARTICLES_ENABLE_HDF5 OFF)
option(NESO_PARTICLES_ENABLE_TESTS OFF)
option(NESO_PARTICLES_ENABLE_FIND_SYCL ON)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/NESO-Particles)

find_sycl_if_required()

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(two_stream src/two_stream.cpp)

target_link_libraries(two_stream PUBLIC NESO-Particles::NESO-Particles)
add_sycl_to_target(TARGET two_stream SOURCES src/two_stream.cpp)

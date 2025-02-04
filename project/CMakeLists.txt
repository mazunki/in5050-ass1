
cmake_minimum_required(VERSION 3.14)

# Project setup
project(codec63-g01
  VERSION 1.0
  DESCRIPTION "CMake configuration file for Codec63 in C"
  LANGUAGES C CUDA
)

set(CMAKE_CUDA_HOST_COMPILER clang)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize")
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)  # <- for speeds

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-Werror -Wno-error=unused-parameter)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_C_COMPILER_ID STREQUAL GNU)
  message(FATAL "don't use gcc please")
endif()


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 -O2")  # Modify sm_50 if needed

include_directories(include)

# common
add_library(c63 STATIC
    src/dsp.c
    src/tables.c
    src/io.c
    src/common.c
    src/me.c
)


# encoder
add_executable(c63enc
    src/c63enc.c
    src/c63_write.c
)
target_link_libraries(c63enc c63 m)

# decoder
add_executable(c63dec
  src/c63dec.c include/c63.h
)
target_link_libraries(c63dec c63 m)


# Prediction mode (optional, used for debugging)
add_executable(c63pred
  src/c63dec.c
)
target_link_libraries(c63pred c63 m)
target_compile_options(c63pred PUBLIC -DC63_PRED)


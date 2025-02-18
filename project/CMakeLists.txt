cmake_minimum_required(VERSION 3.14)

project(Codec63
    VERSION 3.0
    DESCRIPTION "CMake configuration file for Codec63 in CUDA"
    LANGUAGES CUDA C)

# if(NOT CMAKE_C_COMPILER_ID STREQUAL GNU)
#   message(FATAL_ERROR "please compile with gcc")  # <- nvcc works best with gcc
# endif()

# Use CUDA standard (project-wide setting)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
set(CMAKE_CUDA_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)



# Library for shared functionality
add_library(c63 STATIC
    src/quantdct.cu include/quantdct.h
    src/tables.cu include/tables.h
    src/io.cu include/io.h
    src/common.cu include/common.h
    src/me.cu include/me.h
)
# target_include_directories(c63 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_options(c63 PUBLIC -I${CMAKE_SOURCE_DIR}/include)


# Encoder executable
add_executable(c63enc
    src/c63enc.cu include/c63.h
    src/c63_write.cu include/c63_write.h
)
target_link_libraries(c63enc c63 m)

# Decoder executable
add_executable(c63dec
	src/c63dec.cu include/c63.h
)
target_link_libraries(c63dec c63 m)

# Prediction executable
add_executable(c63pred
	src/c63dec.cu include/c63.h
)
target_link_libraries(c63pred c63 m)
target_compile_options(c63pred PUBLIC -DC63_PRED)


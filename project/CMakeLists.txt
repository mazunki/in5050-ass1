cmake_minimum_required(VERSION 3.14)

#
# This tells CMake that your project's name is homeexam-01, and that the programs
# here are written in C.
# 
project(Codec63
	VERSION 3.0
	DESCRIPTION "CMake configuration file for Codec63 in C"
	LANGUAGES CUDA)

# Use the C standard from 2011. No effect on Windows (Visual Studio ignores
# the standards anyway).
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize  -fno-omit-frame-pointer")


# Make sure Debug mode flags are similar to Release mode flags
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_50,code=sm_50")

# CMake configures Makefiles for discrete GPUs, but it doesn't do it
# for integrated GPUs like the Tegra. Pick the right compute capabilities
# explicitly. A bit hackish because a Tegra might have a discrete GPU.
# But it cannot use both at the same time.
if(EXISTS "/etc/nv_tegra_release")
  # The CC list for Tegras and Jetson will require manual updates
  set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
else()
  # The CC list for discrete GPUs will require CMake updates
  set(CMAKE_CUDA_ARCHITECTURES "all-major"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
endif()

# Debug compiles with -g or equivalent for debugging.
# Change to Release for good speed.
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

#
# If this is a Tegra or something similar, set our processor subrevision.
# On Mac, CMake adds arch flags by itself.
#
if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  if(EXISTS "/etc/nv_tegra_release")
    add_compile_options(-march=armv8.2-a)
  endif()
else()
  message(FATAL "Not compiling with gcc")
endif()

# This march option works with GCC but apparently not with the CUDA compiler on Tegra
# if(EXISTS "/etc/nv_tegra_release")
#     add_compile_options(-march=armv8.2-a)
# endif()

include_directories(include)

add_library(c63
            STATIC src/quantdct.cu
	           src/tables.cu
	           src/io.cu
	           src/common.cu
	           src/me.cu)

add_executable(c63enc
               src/c63enc.cu
	       src/c63_write.cu)
target_link_libraries(c63enc c63 m)

add_executable(c63dec
               src/c63dec.cu)
target_link_libraries(c63dec c63 m)

add_executable(c63pred
               src/c63dec.cu)
target_link_libraries(c63pred c63 m)
target_compile_options(c63pred PUBLIC -DC63_PRED)



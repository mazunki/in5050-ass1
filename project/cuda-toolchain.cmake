# host-specific flags

set(CMAKE_C_COMPILER "/usr/bin/gcc-13" CACHE STRING "Use GCC for C compiler")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13" CACHE STRING "Use G++ for C++ compiler")

set(CUDA_HOME "/usr/local/cuda" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-13" CACHE STRING "CUDA host compiler" FORCE)

set(CMAKE_EXE_LINKER_FLAGS "-L/usr/local/cuda/lib64")


# Set CUDA architectures (modify if needed)
if(EXISTS "/etc/nv_tegra_release")
  set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87"
      CACHE STRING "CUDA architectures for Tegra/Jetson")

  set(ARMV8_OPTIMIZATION "-march=armv8.2-a")
  add_compile_options(${ARMV8_OPTIMIZATION})
else()
  set(CMAKE_CUDA_ARCHITECTURES "all-major"
      CACHE STRING "CUDA architectures for discrete GPUs")
endif()

# Ensure environment variables are set for CUDA
set(ENV{CUDA_HOME} "/usr/local/cuda")
set(ENV{PATH} "$ENV{CUDA_HOME}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")


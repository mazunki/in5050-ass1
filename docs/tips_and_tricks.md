# in5050
# home exam 1 tips and tricks

- Make code that has flexible block sizes, grid sizes
- Don't overlook doing stuff on the CPU, maybe it's quicker? Something something synchronization
- 1 Core per SAD => 8 Grid => 8 Block (maybe?, example)
- DCT should be on GPU this assignment

- Pre-compute sad values into cache, maybe SIMD cmp them for equality in cache?
- Can't compare all 256 values (only 22?) with simd-cmp, maybe store a float/hash value instead?
- Convenient fact 8x8 is compared to at most 32x32 other blocks (start at 1024 and then search logarithmically by comparing pairs)
- Validation is done visually, by making sure the foreman isn't pink, and not algorithmically. No deduction for being a few percent off.
- PCNR (?) can be used to compare bytes, but it's shit for comparing video perception
- Dedicated memory locations by pinning things
- `memalign` instead of `malloc`. For this probably best to use CUDA host.
- Instead of `cuda_memcpy`, maybe use streams for asynchronous copy
- Maybe change the order of the loop and collect the results of each of them?
- "Reduction", CUDA, "shuffle"
- Blocks have x,y,z positions. We can think of them as SIMD instructions. We can use shuffling here to run several instructions from a given start with some offset

- `sad_block_8x8` called by motion estimated a bajillion times due to O(n²) probably. find best candidate differently instead

- texture memory space is just a view into the global memory space
- constant memory is tiny

- running each starting I-frame concurrently is unlikely to be beneficial, plus the input should be a stream (not a static file)

- loading 128 bytes at once is bestest

- Parallelize each chroma/luma channel independently, maybe in parallel. Likely to be a good thing

- Build a tree of instruction relationships, parallelize all the loops, figure out the kernel/core/thread/grid/block count

## for the report
- Write down what didn't work for the report
- Add graphs for presentation


#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "me.h"
#include "tables.h"
#include "common.h"

#define MACROBLOCK_SIZE 8
#define CUDA_THREADS_PER_BLOCK_X 32
#define CUDA_THREADS_PER_BLOCK_Y 32

__device__ static int sad_block_8x8(const uint8_t *__restrict__ block1, const uint8_t *__restrict__ block2, int stride)
{
    int u = threadIdx.x;
    int v = threadIdx.y;
    
    int local_sad = 0;
    if (u < MACROBLOCK_SIZE && v < MACROBLOCK_SIZE)
    {
        local_sad = abs(block2[v * stride + u] - block1[v * stride + u]);
    }
    
    // Warp-level reduction for performance
    for (int offset = 4; offset > 0; offset /= 2)
    {
        local_sad += __shfl_down_sync(0xFFFFFFFF, local_sad, offset);
    }
    return local_sad;
}

__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y,
                                    const uint8_t *__restrict__ orig, const uint8_t *__restrict__ ref, 
                                    int padw, int padh, int range)
{
    __shared__ uint8_t shared_ref[MACROBLOCK_SIZE][MACROBLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load reference block into shared memory
    int ref_x = mb_x * MACROBLOCK_SIZE + tx;
    int ref_y = mb_y * MACROBLOCK_SIZE + ty;
    if (tx < MACROBLOCK_SIZE && ty < MACROBLOCK_SIZE)
    {
        shared_ref[ty][tx] = ref[ref_y * padw + ref_x];
    }
    __syncthreads();

    int left   = max(mb_x * MACROBLOCK_SIZE - range, 0);
    int top    = max(mb_y * MACROBLOCK_SIZE - range, 0);
    int right  = min(mb_x * MACROBLOCK_SIZE + range, padw - MACROBLOCK_SIZE);
    int bottom = min(mb_y * MACROBLOCK_SIZE + range, padh - MACROBLOCK_SIZE);

    int best_sad = INT_MAX;
    int best_mv_x = 0, best_mv_y = 0;

    for (int y = top; y < bottom; ++y)
    {
        for (int x = left; x < right; ++x)
        {
            int sad = sad_block_8x8(orig + (mb_y * MACROBLOCK_SIZE) * padw + mb_x * MACROBLOCK_SIZE,
                                    &shared_ref[0][0], MACROBLOCK_SIZE);
            if (sad < best_sad)
            {
                best_sad = sad;
                best_mv_x = x - mb_x * MACROBLOCK_SIZE;
                best_mv_y = y - mb_y * MACROBLOCK_SIZE;
            }
        }
    }

    mb->mv_x = best_mv_x;
    mb->mv_y = best_mv_y;
    mb->use_mv = 1;
}

__global__ void c63_motion_estimate_kernel(uint8_t *__restrict__ d_orig, uint8_t *__restrict__ d_recons, macroblock *__restrict__ d_mbs, int width, int height, int range) {
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    int mb_cols = width / MACROBLOCK_SIZE;
    int mb_rows = height / MACROBLOCK_SIZE;

    if (mb_x >= mb_cols || mb_y >= mb_rows) {
        return;
    }

    macroblock *mb = &d_mbs[mb_y * mb_cols + mb_x];
    me_block_8x8(mb, mb_x, mb_y, d_orig, d_recons, width, height, range);
}

void c63_motion_estimate(struct c63_common *cm)
{
    /* CUDA kernel launch for motion estimation */
    int range = cm->me_search_range;
    dim3 block_size(CUDA_THREADS_PER_BLOCK_X, CUDA_THREADS_PER_BLOCK_Y);
    dim3 grid_size(cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE, cm->padh[Y_COMPONENT] / MACROBLOCK_SIZE);

    c63_motion_estimate_kernel<<<grid_size, block_size>>>(
        cm->curframe->orig->d_Y, cm->refframe->recons->d_Y, cm->curframe->d_mbs[Y_COMPONENT],
        cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], range);
    cudaDeviceSynchronize();
}
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
    /* Motion compensation remains unchanged */
    for (int mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
    {
        for (int mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
        {
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
                          cm->refframe->recons->Y, Y_COMPONENT);
        }
    }
    for (int mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
    {
        for (int mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
        {
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
                          cm->refframe->recons->U, U_COMPONENT);
            mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
                          cm->refframe->recons->V, V_COMPONENT);
        }
    }
}


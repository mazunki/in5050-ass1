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
#include <cuda_runtime.h>

#include "me.h"
#include "tables.h"
#include "common.h"

__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride)
{
  __shared__ int shared_sad[MACROBLOCK_SIZE][MACROBLOCK_SIZE];

  int row = threadIdx.y;
  int col = threadIdx.x;

  if (row < MACROBLOCK_SIZE && col < MACROBLOCK_SIZE)
  {
    shared_sad[row][col] = abs(block1[row * stride + col] - block2[row * stride + col]);
  }
  else
{
    shared_sad[row][col] = 0;
  }

  __syncthreads();

  for (int s = MACROBLOCK_SIZE / 2; s > 0; s /= 2) {
    if (row < s) {
      shared_sad[row][col] += shared_sad[row + s][col];
    }
    __syncthreads();
  }

  return (row == 0 && col == 0) ? shared_sad[0][col] : 0;
}



__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y,
                                    uint8_t *orig, uint8_t *ref, int padw, int padh, int range)
{
  int left   = MAX(mb_x * MACROBLOCK_SIZE - range, 0);
  int top    = MAX(mb_y * MACROBLOCK_SIZE - range, 0);
  int right  = MIN(mb_x * MACROBLOCK_SIZE + range, padw - MACROBLOCK_SIZE);
  int bottom = MIN(mb_y * MACROBLOCK_SIZE + range, padh - MACROBLOCK_SIZE);

  int mx = mb_x * MACROBLOCK_SIZE;
  int my = mb_y * MACROBLOCK_SIZE;

  __shared__ int best_sad;
  __shared__ int best_mv_x;
  __shared__ int best_mv_y;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    best_sad = INT_MAX;
    best_mv_x = 0;
    best_mv_y = 0;
  }
  __syncthreads();

  int local_best_sad = INT_MAX;
  int local_best_x = 0, local_best_y = 0;

  for (int y = top + threadIdx.y; y < bottom; y += blockDim.y) {
    for (int x = left + threadIdx.x; x < right; x += blockDim.x) {
      int sad = sad_block_8x8(orig + my * padw + mx, ref + y * padw + x, padw);
      if (sad < local_best_sad) {
        local_best_sad = sad;
        local_best_x = x - mx;
        local_best_y = y - my;
      }
    }
  }

  __syncthreads();

  // Atomic update only for SAD (not motion vectors)
  if (atomicMin(&best_sad, local_best_sad) > local_best_sad) {
    best_mv_x = local_best_x;
    best_mv_y = local_best_y;
  }

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mb->mv_x = best_mv_x;
    mb->mv_y = best_mv_y;
    mb->use_mv = 1;
  }
}




/**
@param[in] d_orig
@param[in] d_recons (from last frame)
@param[out] d_mbs
*/
__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int width, int height, int range)
{
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;
  int mb_cols = width / MACROBLOCK_SIZE;
  int mb_rows = height / MACROBLOCK_SIZE;

  if (mb_x >= mb_cols || mb_y >= mb_rows) {
    return;
  }

  macroblock *mb = &d_mbs[mb_y * mb_cols + mb_x];
  me_block_8x8(mb, mb_x, mb_y, d_orig, d_recons, width, height, range);
}

__host__ void c63_motion_estimate(struct c63_common *cm)
{
  dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
  dim3 grid_size_luma(cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE, cm->padh[Y_COMPONENT] / MACROBLOCK_SIZE);
  dim3 grid_size_chroma(cm->padw[U_COMPONENT] / MACROBLOCK_SIZE, cm->padh[U_COMPONENT] / MACROBLOCK_SIZE);

  c63_pipeline *pipe = cm->pipe;

  c63_motion_estimate_kernel<<<grid_size_luma, block_size, 0, pipe->stream_estimate>>>(pipe->d_orig_Y, pipe->d_refframe_Y, pipe->d_mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->me_search_range);
  CUDA_ASSERT();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate>>>(pipe->d_orig_U, pipe->d_refframe_U, pipe->d_mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->me_search_range/2);
  CUDA_ASSERT();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate>>>(pipe->d_orig_V, pipe->d_refframe_V, pipe->d_mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->me_search_range/2);
  CUDA_ASSERT();

}




/* Motion compensation for 8x8 block */
/**
@param[in] d_mbs
@param[out] d_predicted
@param[in] d_ref
*/
__global__ void c63_motion_compensate_kernel(macroblock *d_mbs, int mb_cols, int mb_rows,
                                             uint8_t *d_predicted, uint8_t *d_ref, int padw)
{
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  if (mb_x >= mb_cols || mb_y >= mb_rows) return;

  macroblock *mb = &d_mbs[mb_y * mb_cols + mb_x];
  if (!mb->use_mv) return;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int left = mb_x * MACROBLOCK_SIZE;
  int top = mb_y * MACROBLOCK_SIZE;

  __shared__ uint8_t pred_block[MACROBLOCK_SIZE][MACROBLOCK_SIZE];

  pred_block[ty][tx] = d_ref[(top + ty + mb->mv_y) * padw + (left + tx + mb->mv_x)];
  __syncthreads();

  d_predicted[(top + ty) * padw + (left + tx)] = pred_block[ty][tx];
}


__host__ void c63_motion_compensate_cuda(struct c63_common *cm)
{
  dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
  dim3 grid_size_luma(cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE, cm->padh[Y_COMPONENT] / MACROBLOCK_SIZE);
  dim3 grid_size_chroma(cm->padw[U_COMPONENT] / MACROBLOCK_SIZE, cm->padh[U_COMPONENT] / MACROBLOCK_SIZE);

  c63_pipeline *pipe = cm->pipe;

  /* Luma */
  c63_motion_compensate_kernel<<<grid_size_luma, block_size, 0, pipe->stream_compensate>>>(pipe->d_mbs[Y_COMPONENT], cm->mb_cols, cm->mb_rows, pipe->d_predicted_Y, pipe->d_refframe_Y, cm->padw[Y_COMPONENT]);
  CUDA_ASSERT();

  /* Chroma */
  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate>>>(pipe->d_mbs[U_COMPONENT], cm->mb_cols/2, cm->mb_rows/2, pipe->d_predicted_U, pipe->d_refframe_U, cm->padw[U_COMPONENT]);
  CUDA_ASSERT();

  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate>>>(pipe->d_mbs[V_COMPONENT], cm->mb_cols/2, cm->mb_rows/2, pipe->d_predicted_V, pipe->d_refframe_V, cm->padw[V_COMPONENT]);
  CUDA_ASSERT();
}



// non-cuda version used by decoder
static void mc_block_8x8(struct macroblock *mb, int mb_x, int mb_y,
                         uint8_t *predicted, uint8_t *ref, int padw)
{
  if (!mb->use_mv) { return; }

  int left = mb_x * MACROBLOCK_SIZE;
  int top = mb_y * MACROBLOCK_SIZE;
  int right = left + MACROBLOCK_SIZE;
  int bottom = top + MACROBLOCK_SIZE;
  int w = padw;

  for (int y = top; y < bottom; ++y)
  {
    for (int x = left; x < right; ++x)
    {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      struct macroblock *mb = &cm->curframe->mbs[Y_COMPONENT][mb_y * (cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, cm->padw[Y_COMPONENT]);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      struct macroblock *mb_u = &cm->curframe->mbs[U_COMPONENT][mb_y * (cm->padw[U_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb_u, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, cm->padw[U_COMPONENT]);

      struct macroblock *mb_v = &cm->curframe->mbs[V_COMPONENT][mb_y * (cm->padw[V_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb_v, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, cm->padw[V_COMPONENT]);
    }
  }
}


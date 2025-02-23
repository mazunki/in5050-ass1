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

#include "me.h"
#include "tables.h"
#include "common.h"

#define MACROBLOCK_SIZE 8
#define CUDA_THREADS_PER_BLOCK_X 16
#define CUDA_THREADS_PER_BLOCK_Y 16

__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride)
{
  int u, v;
  int result = 0;
  for (v = 0; v < MACROBLOCK_SIZE; ++v)
  {
    for (u = 0; u < MACROBLOCK_SIZE; ++u)
    {
      result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
  return result;
}

/* Motion estimation for an 8x8 block */
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y,
                                    uint8_t *orig, uint8_t *ref, int padw, int padh, int range)
{
  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  int left   = MAX(mb_x * MACROBLOCK_SIZE - range, 0);
  int top    = MAX(mb_y * MACROBLOCK_SIZE - range, 0);
  int right  = MIN(mb_x * MACROBLOCK_SIZE + range, padw - MACROBLOCK_SIZE);
  int bottom = MIN(mb_y * MACROBLOCK_SIZE + range, padh - MACROBLOCK_SIZE);

  int x, y;
  int mx = mb_x * MACROBLOCK_SIZE;
  int my = mb_y * MACROBLOCK_SIZE;
  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad = sad_block_8x8(orig + my*padw + mx, ref + y*padw + x, padw);
      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int width, int height, int range) {
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

__host__ void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int range = cm->me_search_range;
  size_t frame_size = cm->ypw * cm->yph;
  size_t chroma_size = (cm->ypw / 2) * (cm->yph / 2);
  size_t num_blocks_luma = cm->mb_rows * cm->mb_cols;
  size_t num_blocks_chroma = (cm->mb_rows / 2) * (cm->mb_cols / 2);

  dim3 block_size(CUDA_THREADS_PER_BLOCK_X, CUDA_THREADS_PER_BLOCK_Y);
  dim3 grid_size(cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE, cm->padh[Y_COMPONENT] / MACROBLOCK_SIZE);

  CUDA_CHECK(cudaMemcpy(cm->curframe->orig->d_Y, cm->curframe->orig->Y, frame_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cm->curframe->orig->d_U, cm->curframe->orig->U, chroma_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cm->curframe->orig->d_V, cm->curframe->orig->V, chroma_size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(cm->refframe->recons->d_Y, cm->refframe->recons->Y, frame_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cm->refframe->recons->d_U, cm->refframe->recons->U, chroma_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cm->refframe->recons->d_V, cm->refframe->recons->V, chroma_size, cudaMemcpyHostToDevice));

  /* Luma */
  c63_motion_estimate_kernel<<<grid_size, block_size>>>(cm->curframe->orig->d_Y, cm->refframe->recons->d_Y, cm->curframe->d_mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], range);
  CUDA_ASSERT();

  /* Chroma */
  c63_motion_estimate_kernel<<<grid_size, block_size>>>(cm->curframe->orig->d_U, cm->refframe->recons->d_U, cm->curframe->d_mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], range/2);
  CUDA_ASSERT();

  c63_motion_estimate_kernel<<<grid_size, block_size>>>(cm->curframe->orig->d_V, cm->refframe->recons->d_V, cm->curframe->d_mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], range/2);
  CUDA_ASSERT();

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], cm->curframe->d_mbs[Y_COMPONENT], num_blocks_luma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[U_COMPONENT], cm->curframe->d_mbs[U_COMPONENT], num_blocks_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[V_COMPONENT], cm->curframe->d_mbs[V_COMPONENT], num_blocks_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
}

/* Motion compensation for 8x8 block */
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

/* Motion compensation kernel function (still CPU-based) */
static void c63_motion_compensate_kernel(struct macroblock *mbs, int mb_cols, int mb_rows,
                                         uint8_t *predicted, uint8_t *ref, int padw)
{
  for (int mb_y = 0; mb_y < mb_rows; ++mb_y)
  {
    for (int mb_x = 0; mb_x < mb_cols; ++mb_x)
    {
      struct macroblock *mb = &mbs[mb_y * mb_cols + mb_x];

      mc_block_8x8(mb, mb_x, mb_y, predicted, ref, padw);
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  /* Luma */
  c63_motion_compensate_kernel(cm->curframe->mbs[Y_COMPONENT], cm->mb_cols, cm->mb_rows, cm->curframe->predicted->Y, cm->refframe->recons->Y, cm->padw[Y_COMPONENT]);

  /* Chroma */
  c63_motion_compensate_kernel(cm->curframe->mbs[U_COMPONENT], cm->mb_cols / 2, cm->mb_rows / 2, cm->curframe->predicted->U, cm->refframe->recons->U, cm->padw[U_COMPONENT]);

  c63_motion_compensate_kernel(cm->curframe->mbs[V_COMPONENT], cm->mb_cols / 2, cm->mb_rows / 2, cm->curframe->predicted->V, cm->refframe->recons->V, cm->padw[V_COMPONENT]);
}

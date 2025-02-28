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

#include "common.h"
#include "me.h"
#include "tables.h"


// estimation
__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int width, int height, int range);

__device__ static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result);
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range);


// compensation
__global__ void c63_motion_compensate_kernel(macroblock *d_mbs, int mb_cols, int mb_rows, uint8_t *d_predicted, uint8_t *d_ref, int padw);


__host__ __device__ static void mc_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int padw);



/**
 * @brief Motion estimation
 *
 * Motion estimation calculates the motion vectors using
 * the original image from a reference image (made by
 * reconstructing the previous frame
 *
 * This is only used during encoding, since the decoder only
 * has the original image during key-frames.
 *
 * @param[in]  d_orig
 * @param[in]  d_recons
 * @param[out] d_mbs
 */
__host__ void c63_motion_estimate(struct c63_common *cm)
{
  dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
  dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
  dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

  c63_pipeline *pipe = cm->pipe;

  c63_motion_estimate_kernel<<<grid_size_luma,   block_size>>>(pipe->d_orig_Y, pipe->d_refframe_Y, pipe->d_mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->me_search_range);
  CUDA_CHECK();
  c63_motion_estimate_kernel<<<grid_size_chroma, block_size>>>(pipe->d_orig_U, pipe->d_refframe_U, pipe->d_mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->me_search_range/2);
  CUDA_CHECK();
  c63_motion_estimate_kernel<<<grid_size_chroma, block_size>>>(pipe->d_orig_V, pipe->d_refframe_V, pipe->d_mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->me_search_range/2);
  CUDA_CHECK();

}

/**
 * @brief Sums up the Sum of Absolute Difference between two blocks.
 * 
 * This value can then be used to pick the best match for any given
 * macroblock during motion estimation.
 */
__device__ static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < MACROBLOCK_SIZE; ++v)
  {
    for (u = 0; u < MACROBLOCK_SIZE; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
}

/* performs motion estimation for a full macroblock */
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range)
{
  int left = mb_x * MACROBLOCK_SIZE - range;
  int top = mb_y * MACROBLOCK_SIZE - range;
  int right = mb_x * MACROBLOCK_SIZE + range;
  int bottom = mb_y * MACROBLOCK_SIZE + range;

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (padw - MACROBLOCK_SIZE)) { right = padw - MACROBLOCK_SIZE; }
  if (bottom > (padh - MACROBLOCK_SIZE)) { bottom = padh - MACROBLOCK_SIZE; }

  int x, y;

  int mx = mb_x * MACROBLOCK_SIZE;
  int my = mb_y * MACROBLOCK_SIZE;

  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad;
      sad_block_8x8(orig + my*padw+mx, ref + y*padw+x, padw, &sad);

      /* DEBUG("(%4d,%4d) - %d", x, y, sad); */

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

  /* DEBUG("Using motion vector (%d, %d) with SAD %d", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

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


/**
 * @brief Motion Compensation
 * 
 * Motion compensation predicts what a frame would look like
 * using the datablocks provided to it, and the previous
 * frame's reconstructed frame.
 *
 * This is used both during encoding and decoding.
 *
 * @param[in]  d_mbs
 * @param[out] d_predicted
 * @param[in]  d_ref
 */
__host__ void c63_motion_compensate(struct c63_common *cm)
{
  dim3 block_size(MACROBLOCK_SIZE, MACROBLOCK_SIZE);
  dim3 grid_size_luma(cm->mb_cols_luma, cm->mb_rows_luma);
  dim3 grid_size_chroma(cm->mb_cols_chroma, cm->mb_rows_chroma);

  c63_pipeline *pipe = cm->pipe;

  c63_motion_compensate_kernel<<<grid_size_luma,   block_size>>>(pipe->d_mbs[Y_COMPONENT], cm->mb_cols_luma,   cm->mb_rows_luma,   pipe->d_predicted_Y, pipe->d_refframe_Y, cm->padw[Y_COMPONENT]);
  CUDA_CHECK();
  c63_motion_compensate_kernel<<<grid_size_chroma, block_size>>>(pipe->d_mbs[U_COMPONENT], cm->mb_cols_chroma, cm->mb_rows_chroma, pipe->d_predicted_U, pipe->d_refframe_U, cm->padw[U_COMPONENT]);
  CUDA_CHECK();
  c63_motion_compensate_kernel<<<grid_size_chroma, block_size>>>(pipe->d_mbs[V_COMPONENT], cm->mb_cols_chroma, cm->mb_rows_chroma, pipe->d_predicted_V, pipe->d_refframe_V, cm->padw[V_COMPONENT]);
  CUDA_CHECK();
}

__global__ void c63_motion_compensate_kernel(struct macroblock *mbs, int mb_cols, int mb_rows, uint8_t *predicted, uint8_t *ref, int padw)
{
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  if (mb_x >= mb_cols || mb_y >= mb_rows) return;

  struct macroblock *mb = &mbs[mb_y * mb_cols + mb_x];
  mc_block_8x8(mb, mb_x, mb_y, predicted, ref, padw);
}

void c63_motion_compensate_legacy(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows_luma; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols_chroma; ++mb_x)
    {
      struct macroblock *mb = &cm->curframe->mbs[Y_COMPONENT] [mb_y * cm->mb_cols_luma + mb_x];
      mc_block_8x8(mb, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, cm->padw[Y_COMPONENT]);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows_chroma; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols_chroma; ++mb_x)
    {
      struct macroblock *mb_u = &cm->curframe->mbs[U_COMPONENT][mb_y * cm->mb_cols_luma + mb_x];
      mc_block_8x8(mb_u, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, cm->padw[U_COMPONENT]);

      struct macroblock *mb_v = &cm->curframe->mbs[V_COMPONENT][mb_y * cm->mb_cols_luma + mb_x];
      mc_block_8x8(mb_v, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, cm->padw[V_COMPONENT]);
    }
  }
}

/* writes the prediction for a full macroblock */
__host__ __device__ static void mc_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int padw)
{
  if (!mb->use_mv) { return; }

  int left = mb_x * MACROBLOCK_SIZE;
  int top = mb_y * MACROBLOCK_SIZE;
  int right = left + MACROBLOCK_SIZE;
  int bottom = top + MACROBLOCK_SIZE;

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*padw+x] = ref[(y + mb->mv_y) * padw + (x + mb->mv_x)];
    }
  }
}


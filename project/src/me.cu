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

__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride);
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range);


// compensation
__global__ void c63_motion_compensate_kernel(macroblock *d_mbs, int mb_cols, int mb_rows, uint8_t *d_predicted, uint8_t *d_ref, int padw);



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

  CUDA_ASSERT(cudaStreamSynchronize(pipe->stream_image));

  c63_motion_estimate_kernel<<<grid_size_luma,   block_size, 0, pipe->stream_estimate_Y>>>(pipe->d_orig_Y, pipe->d_refframe_Y, pipe->d_mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->me_search_range);
  CUDA_CHECK();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_U>>>(pipe->d_orig_U, pipe->d_refframe_U, pipe->d_mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->me_search_range/2);
  CUDA_CHECK();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_V>>>(pipe->d_orig_V, pipe->d_refframe_V, pipe->d_mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->me_search_range/2);
  CUDA_CHECK();

}

/**
 * @brief Sums up the Sum of Absolute Difference between two blocks.
 * 
 * This value can then be used to pick the best match for any given
 * macroblock during motion estimation.
 */
__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride)
{
  __shared__ int s_sad_values[MACROBLOCK_SIZE][MACROBLOCK_SIZE];

  int row = threadIdx.y;
  int col = threadIdx.x;

  if (row < MACROBLOCK_SIZE && col < MACROBLOCK_SIZE)
  {
    s_sad_values[row][col] = abs(block1[row * stride + col] - block2[row * stride + col]);
  }
  else
{
    s_sad_values[row][col] = 0;
  }

  __syncthreads();

  for (int s = MACROBLOCK_SIZE / 2; s > 0; s /= 2) {
    if (row < s) {
      s_sad_values[row][col] += s_sad_values[row + s][col];
    }
    __syncthreads();
  }

  return (row == 0 && col == 0) ? s_sad_values[0][col] : 0;
}


/* performs motion estimation for a full macroblock */
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range)
{
  int left   = MAX(mb_x * MACROBLOCK_SIZE - range, 0);
  int top    = MAX(mb_y * MACROBLOCK_SIZE - range, 0);
  int right  = MIN(mb_x * MACROBLOCK_SIZE + range, padw - MACROBLOCK_SIZE);
  int bottom = MIN(mb_y * MACROBLOCK_SIZE + range, padh - MACROBLOCK_SIZE);

  int mx = mb_x * MACROBLOCK_SIZE;
  int my = mb_y * MACROBLOCK_SIZE;

  __shared__ int s_best_sad;
  __shared__ int s_best_mv_x;
  __shared__ int s_best_mv_y;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    s_best_sad = INT_MAX;
    s_best_mv_x = 0;
    s_best_mv_y = 0;
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

  if (atomicMin(&s_best_sad, local_best_sad) > local_best_sad) {
    s_best_mv_x = local_best_x;
    s_best_mv_y = local_best_y;
  }

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    mb->mv_x = s_best_mv_x;
    mb->mv_y = s_best_mv_y;
    mb->use_mv = 1;
  }
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

  CUDA_ASSERT(cudaStreamSynchronize(pipe->stream_estimate_Y));

  c63_motion_compensate_kernel<<<grid_size_luma,   block_size, 0, pipe->stream_compensate_Y>>>(pipe->d_mbs[Y_COMPONENT], cm->mb_cols_luma,   cm->mb_rows_luma,   pipe->d_predicted_Y, pipe->d_refframe_Y, cm->padw[Y_COMPONENT]);
  CUDA_CHECK();

  CUDA_ASSERT(cudaStreamSynchronize(pipe->stream_estimate_U));
  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate_U>>>(pipe->d_mbs[U_COMPONENT], cm->mb_cols_chroma, cm->mb_rows_chroma, pipe->d_predicted_U, pipe->d_refframe_U, cm->padw[U_COMPONENT]);
  

  CUDA_ASSERT(cudaStreamSynchronize(pipe->stream_estimate_V));
  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate_V>>>(pipe->d_mbs[V_COMPONENT], cm->mb_cols_chroma, cm->mb_rows_chroma, pipe->d_predicted_V, pipe->d_refframe_V, cm->padw[V_COMPONENT]);
  CUDA_CHECK();
}

__global__ void c63_motion_compensate_kernel(struct macroblock *d_mbs, int mb_cols, int mb_rows, uint8_t *d_predicted, uint8_t *d_ref, int padw)
{
  __shared__ uint8_t s_predictions[MACROBLOCK_SIZE][MACROBLOCK_SIZE];

  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;

  if (mb_x >= mb_cols || mb_y >= mb_rows) return;

  macroblock *mb = &d_mbs[mb_y * mb_cols + mb_x];
  if (!mb->use_mv) return;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int left = mb_x * MACROBLOCK_SIZE;
  int top = mb_y * MACROBLOCK_SIZE;

  s_predictions[ty][tx] = d_ref[(top + ty + mb->mv_y) * padw + (left + tx + mb->mv_x)];
  __syncthreads();

  d_predicted[(top + ty) * padw + (left + tx)] = s_predictions[ty][tx];
}


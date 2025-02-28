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
__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int comp);

__device__ static int sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride);
__device__ static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range);


// compensation
__global__ void c63_motion_compensate_kernel(struct macroblock *d_mbs, uint8_t *d_predicted, uint8_t *d_ref, int comp);

/** constant memory */
__constant__ int c_padw[COLOR_COMPONENTS];
__constant__ int c_padh[COLOR_COMPONENTS];
__constant__ int c_mb_cols[COLOR_COMPONENTS];
__constant__ int c_mb_rows[COLOR_COMPONENTS];
__constant__ int c_me_search_range;

void c63_initialize_constant_values(struct c63_common *cm)
{
    int padw[3] = {cm->padw[Y_COMPONENT], cm->padw[U_COMPONENT], cm->padw[V_COMPONENT]};
    int padh[3] = {cm->padh[Y_COMPONENT], cm->padh[U_COMPONENT], cm->padh[V_COMPONENT]};
    int mb_cols[3] = {cm->mb_cols_luma, cm->mb_cols_chroma, cm->mb_cols_chroma};
    int mb_rows[3] = {cm->mb_rows_luma, cm->mb_rows_chroma, cm->mb_rows_chroma};
    int me_range = cm->me_search_range;

    cudaMemcpyToSymbol(c_padw, padw, sizeof(padw));
    cudaMemcpyToSymbol(c_padh, padh, sizeof(padh));
    cudaMemcpyToSymbol(c_mb_cols, mb_cols, sizeof(mb_cols));
    cudaMemcpyToSymbol(c_mb_rows, mb_rows, sizeof(mb_rows));
    cudaMemcpyToSymbol(c_me_search_range, &me_range, sizeof(int));
}



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

  c63_motion_estimate_kernel<<<grid_size_luma, block_size, 0, pipe->stream_estimate_Y>>>(pipe->d_orig_Y, pipe->d_refframe_Y, pipe->d_mbs[Y_COMPONENT], Y_COMPONENT);
  CUDA_CHECK();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_U>>>(pipe->d_orig_U, pipe->d_refframe_U, pipe->d_mbs[U_COMPONENT], U_COMPONENT);
  CUDA_CHECK();

  c63_motion_estimate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_estimate_V>>>(pipe->d_orig_V, pipe->d_refframe_V, pipe->d_mbs[V_COMPONENT], V_COMPONENT);
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


__global__ void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int comp)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    if (mb_x >= c_mb_cols[comp] || mb_y >= c_mb_rows[comp]) return;

    macroblock *mb = &d_mbs[mb_y * c_mb_cols[comp] + mb_x];

    me_block_8x8(mb, mb_x, mb_y, d_orig, d_recons, c_padw[comp], c_padh[comp], c_me_search_range);
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

  c63_motion_compensate_kernel<<<grid_size_luma, block_size, 0, pipe->stream_compensate_Y>>>(pipe->d_mbs[Y_COMPONENT], pipe->d_predicted_Y, pipe->d_refframe_Y, Y_COMPONENT);
  CUDA_CHECK();

  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate_U>>>(pipe->d_mbs[U_COMPONENT], pipe->d_predicted_U, pipe->d_refframe_U, U_COMPONENT);
  CUDA_CHECK();

  c63_motion_compensate_kernel<<<grid_size_chroma, block_size, 0, pipe->stream_compensate_V>>>(pipe->d_mbs[V_COMPONENT], pipe->d_predicted_V, pipe->d_refframe_V, V_COMPONENT);
  CUDA_CHECK();

}

__global__ void c63_motion_compensate_kernel(struct macroblock *d_mbs, uint8_t *d_predicted, uint8_t *d_ref, int comp)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;

    if (mb_x >= c_mb_cols[comp] || mb_y >= c_mb_rows[comp]) return;

    macroblock *mb = &d_mbs[mb_y * c_mb_cols[comp] + mb_x];
    if (!mb->use_mv) return;

    int left = mb_x * MACROBLOCK_SIZE;
    int top = mb_y * MACROBLOCK_SIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    d_predicted[(top + ty) * c_padw[comp] + (left + tx)] = d_ref[(top + ty + mb->mv_y) * c_padw[comp] + (left + tx + mb->mv_x)];
}


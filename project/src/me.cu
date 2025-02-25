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
  dim3 block_size(CUDA_THREADS_PER_BLOCK_X, CUDA_THREADS_PER_BLOCK_Y);
  dim3 grid_size(cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE, cm->padh[Y_COMPONENT] / MACROBLOCK_SIZE);

  c63_pipeline *pipe = cm->pipe;

  CUDA_CHECK(cudaMemcpy(pipe->d_orig_Y, pipe->input->h_orig_Y, cm->frame_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pipe->d_orig_U, pipe->input->h_orig_U, cm->chroma_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pipe->d_orig_V, pipe->input->h_orig_V, cm->chroma_size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(pipe->d_refframe_Y, pipe->input->h_refframe_Y, cm->frame_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pipe->d_refframe_U, pipe->input->h_refframe_U, cm->chroma_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pipe->d_refframe_V, pipe->input->h_refframe_V, cm->chroma_size, cudaMemcpyHostToDevice));


  /* Luma */
  c63_motion_estimate_kernel<<<grid_size, block_size>>>(pipe->d_orig_Y, pipe->d_recons_Y, cm->pipe->d_mbs[Y_COMPONENT], cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->me_search_range);
  CUDA_ASSERT();

  /* Chroma */
  c63_motion_estimate_kernel<<<grid_size, block_size>>>(pipe->d_orig_U, pipe->d_recons_U, cm->pipe->d_mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->me_search_range/2);
  CUDA_ASSERT();

  c63_motion_estimate_kernel<<<grid_size, block_size>>>(pipe->d_orig_V, pipe->d_recons_V, cm->pipe->d_mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->me_search_range/2);
  CUDA_ASSERT();

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[Y_COMPONENT], pipe->d_mbs[Y_COMPONENT], cm->macroblock_count * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[U_COMPONENT], pipe->d_mbs[U_COMPONENT], cm->macroblock_count * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[V_COMPONENT], pipe->d_mbs[V_COMPONENT], cm->macroblock_count * sizeof(struct macroblock), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDeviceSynchronize());
  for (int i=0; i<10; ++i) {
    DEBUG("MV Y[%d]: (%d, %d)", i, pipe->output->h_mbs[Y_COMPONENT][i].mv_x, pipe->output->h_mbs[Y_COMPONENT][i].mv_y);
    DEBUG("MV U[%d]: (%d, %d)", i, pipe->output->h_mbs[U_COMPONENT][i].mv_x, pipe->output->h_mbs[U_COMPONENT][i].mv_y);
    DEBUG("MV V[%d]: (%d, %d)", i, pipe->output->h_mbs[V_COMPONENT][i].mv_x, pipe->output->h_mbs[V_COMPONENT][i].mv_y);
  }
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

void c63_motion_compensate_cuda(struct c63_common *cm)
{
  int mb_x, mb_y;
  c63_pipeline *pipe = cm->pipe;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      struct macroblock *mb = &cm->curframe->mbs[Y_COMPONENT][mb_y * (cm->padw[Y_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb, mb_x, mb_y, pipe->output->h_predicted_Y, pipe->h_recons->Y, cm->padw[Y_COMPONENT]);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      struct macroblock *mb_u = &cm->curframe->mbs[U_COMPONENT][mb_y * (cm->padw[U_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb_u, mb_x, mb_y, pipe->output->h_predicted_U, pipe->h_recons->U, cm->padw[U_COMPONENT]);

      struct macroblock *mb_v = &cm->curframe->mbs[V_COMPONENT][mb_y * (cm->padw[V_COMPONENT] / MACROBLOCK_SIZE) + mb_x];
      mc_block_8x8(mb_v, mb_x, mb_y, pipe->output->h_predicted_V, pipe->h_recons->V, cm->padw[V_COMPONENT]);
    }
  }


  // CUDA_CHECK(cudaMemcpy(pipe->output->h_residuals_Y, pipe->d_residuals_Y, cm->frame_size, cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaMemcpy(pipe->output->h_residuals_U, pipe->d_residuals_U, cm->chroma_size, cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaMemcpy(pipe->output->h_residuals_V, pipe->d_residuals_V, cm->chroma_size, cudaMemcpyDeviceToHost));

  // CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[Y_COMPONENT], pipe->d_mbs[Y_COMPONENT], cm->macroblock_count * sizeof(macroblock), cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[U_COMPONENT], pipe->d_mbs[U_COMPONENT], cm->macroblock_count * sizeof(macroblock), cudaMemcpyDeviceToHost));
  // CUDA_CHECK(cudaMemcpy(pipe->output->h_mbs[V_COMPONENT], pipe->d_mbs[V_COMPONENT], cm->macroblock_count * sizeof(macroblock), cudaMemcpyDeviceToHost));

  // CUDA_CHECK(cudaDeviceSynchronize());

  for (int i=0; i<10; ++i) {
    DEBUG("predicted [%d]: (%d, %d, %d)", i, pipe->output->h_predicted_Y[i], pipe->output->h_predicted_Y[i], pipe->output->h_predicted_Y[i]);
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


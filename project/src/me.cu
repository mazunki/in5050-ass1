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

__device__ void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
    int u, v;
    *result = 0;
    for (v = 0; v < 8; ++v)
    {
        for (u = 0; u < 8; ++u)
        {
            *result += abs(block2[v * stride + u] - block1[v * stride + u]);
        }
    }
}

__device__ void me_block_8x8(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mb, int mb_x, int mb_y, int width, int height)
{
    int range = 16; // Assuming a fixed search range
    int left = mb_x * 8 - range;
    int top = mb_y * 8 - range;
    int right = mb_x * 8 + range;
    int bottom = mb_y * 8 + range;

    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (right > (width - 8)) right = width - 8;
    if (bottom > (height - 8)) bottom = height - 8;

    int mx = mb_x * 8;
    int my = mb_y * 8;
    int best_sad = INT_MAX;
    int best_mv_x = 0, best_mv_y = 0;

    for (int y = top; y < bottom; ++y)
    {
        for (int x = left; x < right; ++x)
        {
            int sad;
            sad_block_8x8(d_orig + my * width + mx, d_recons + y * width + x, width, &sad);
            if (sad < best_sad)
            {
                best_mv_x = x - mx;
                best_mv_y = y - my;
                best_sad = sad;
            }
        }
    }

    d_mb->mv_x = best_mv_x;
    d_mb->mv_y = best_mv_y;
    d_mb->use_mv = 1;
}

__global__ void c63_motion_estimate_kernel(
    uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int width, int height)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (mb_x < width / 8 && mb_y < height / 8) {
        int mb_index = mb_y * (width / 8) + mb_x;
        me_block_8x8(d_orig, d_recons, &d_mbs[mb_index], mb_x, mb_y, width, height);
    }
}

void c63_motion_estimate(struct c63_common *cm)
{
    int width = cm->ypw;
    int height = cm->yph;
    dim3 blockSize(8, 8);
    dim3 gridSize(width / 8, height / 8);

    macroblock *d_mbs;
    cudaMalloc((void **)&d_mbs, (width / 8) * (height / 8) * sizeof(macroblock));

    cudaMemcpy(cm->curframe->recons->d_Y, cm->curframe->recons->Y, width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(cm->curframe->orig->d_Y, cm->curframe->orig->Y, width * height, cudaMemcpyHostToDevice);

    c63_motion_estimate_kernel<<<gridSize, blockSize>>>(cm->curframe->orig->d_Y, cm->curframe->recons->d_Y, d_mbs, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], d_mbs, (width / 8) * (height / 8) * sizeof(macroblock), cudaMemcpyDeviceToHost);
    cudaFree(d_mbs);
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
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}


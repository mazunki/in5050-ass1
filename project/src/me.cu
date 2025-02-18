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

#include <cuda_runtime.h>
#include <stdio.h>
#include <limits.h>

#define BLOCK_SIZE 16  // Define block size for CUDA threads

__device__ int sad_block_8x8(uint8_t *orig, uint8_t *ref, int w) {
    int sad = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            sad += abs(orig[i * w + j] - ref[i * w + j]);
        }
    }
    return sad;
}

__global__ void me_block_8x8_cuda(uint8_t *orig, uint8_t *ref, int w, int h,
                                  int mb_x, int mb_y, int range, int *best_mv_x, int *best_mv_y) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;  // Candidate motion vector X
    int ty = threadIdx.y + blockIdx.y * blockDim.y;  // Candidate motion vector Y

    if (tx >= range * 2 || ty >= range * 2) return; // Ensure we stay within bounds

    int left = max(0, mb_x * 8 - range);
    int top = max(0, mb_y * 8 - range);
    int right = min(w - 8, mb_x * 8 + range);
    int bottom = min(h - 8, mb_y * 8 + range);

    if (tx + left >= right || ty + top >= bottom) return; // Out of bounds check

    int my = mb_y * 8;
    int mx = mb_x * 8;

    int sad = sad_block_8x8(orig + my * w + mx, ref + (ty + top) * w + (tx + left), w);

    __shared__ int best_sad[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int best_x[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int best_y[BLOCK_SIZE][BLOCK_SIZE];

    best_sad[threadIdx.y][threadIdx.x] = sad;
    best_x[threadIdx.y][threadIdx.x] = tx + left - mx;
    best_y[threadIdx.y][threadIdx.x] = ty + top - my;

    __syncthreads();

    // Reduction to find minimum SAD
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int min_sad = INT_MAX;
        int min_x = 0, min_y = 0;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                if (best_sad[i][j] < min_sad) {
                    min_sad = best_sad[i][j];
                    min_x = best_x[i][j];
                    min_y = best_y[i][j];
                }
            }
        }
        atomicMin(best_mv_x, min_x);
        atomicMin(best_mv_y, min_y);
    }
}

void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
    int range = cm->me_search_range;
    if (color_component > 0) { range /= 2; }

    int w = cm->padw[color_component];
    int h = cm->padh[color_component];

    int *d_best_mv_x, *d_best_mv_y;
    cudaMalloc(&d_best_mv_x, sizeof(int));
    cudaMalloc(&d_best_mv_y, sizeof(int));
    cudaMemset(d_best_mv_x, 0, sizeof(int));
    cudaMemset(d_best_mv_y, 0, sizeof(int));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((range * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (range * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    me_block_8x8_cuda<<<gridSize, blockSize>>>(orig, ref, w, h, mb_x, mb_y, range, d_best_mv_x, d_best_mv_y);

    int best_mv_x, best_mv_y;
    cudaMemcpy(&best_mv_x, d_best_mv_x, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best_mv_y, d_best_mv_y, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_best_mv_x);
    cudaFree(d_best_mv_y);

    struct macroblock *mb =
        &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

    mb->mv_x = best_mv_x;
    mb->mv_y = best_mv_y;
    mb->use_mv = 1;
}


void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
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

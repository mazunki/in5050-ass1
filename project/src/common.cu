#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "common.h"

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  frame *f = (frame*)malloc(sizeof(struct frame));
  if (f == NULL)
  {
    return NULL;
  }

  size_t frame_size = cm->ypw * cm->yph;
  size_t chroma_size = (cm->ypw/2) * (cm->yph/2);
  size_t num_blocks_luma = cm->mb_rows * cm->mb_cols;
  size_t num_blocks_chroma = (cm->mb_rows/2) * (cm->mb_cols/2);

  f->orig = image;

  // cpu
  f->recons = (yuv_t*)malloc(sizeof(yuv_t));
  f->recons->Y = (uint8_t*)malloc(frame_size);
  f->recons->U = (uint8_t*)malloc(chroma_size);
  f->recons->V = (uint8_t*)malloc(chroma_size);

  f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
  f->predicted->Y = (uint8_t*)calloc(frame_size, sizeof(uint8_t));
  f->predicted->U = (uint8_t*)calloc(chroma_size, sizeof(uint8_t));
  f->predicted->V = (uint8_t*)calloc(chroma_size, sizeof(uint8_t));

  f->residuals = (dct_t*)malloc(sizeof(dct_t));
  f->residuals->Ydct = (int16_t*)calloc(frame_size, sizeof(int16_t));
  f->residuals->Udct = (int16_t*)calloc(chroma_size, sizeof(int16_t));
  f->residuals->Vdct = (int16_t*)calloc(chroma_size, sizeof(int16_t));

  f->mbs[Y_COMPONENT] = (macroblock*)calloc(num_blocks_luma, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] = (macroblock*)calloc(num_blocks_chroma, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] = (macroblock*)calloc(num_blocks_chroma, sizeof(struct macroblock));

  // gpu
  CUDA_CHECK(cudaMalloc(&f->recons->d_Y, frame_size));
  CUDA_CHECK(cudaMalloc(&f->recons->d_U, chroma_size));
  CUDA_CHECK(cudaMalloc(&f->recons->d_V, chroma_size));

  CUDA_CHECK(cudaMalloc(&f->predicted->d_Y, frame_size));
  CUDA_CHECK(cudaMalloc(&f->predicted->d_U, chroma_size));
  CUDA_CHECK(cudaMalloc(&f->predicted->d_V, chroma_size));

  CUDA_CHECK(cudaMalloc(&f->residuals->d_Ydct, frame_size * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc(&f->residuals->d_Udct, chroma_size * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc(&f->residuals->d_Vdct, chroma_size * sizeof(int16_t)));

  CUDA_CHECK(cudaMalloc(&f->d_mbs[Y_COMPONENT], num_blocks_luma * sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc(&f->d_mbs[U_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc(&f->d_mbs[V_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));

  return f;
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  // gpu
  CUDA_CHECK(cudaFree(f->recons->d_Y));
  CUDA_CHECK(cudaFree(f->recons->d_U));
  CUDA_CHECK(cudaFree(f->recons->d_V));

  CUDA_CHECK(cudaFree(f->predicted->d_Y));
  CUDA_CHECK(cudaFree(f->predicted->d_U));
  CUDA_CHECK(cudaFree(f->predicted->d_V));

  CUDA_CHECK(cudaFree(f->residuals->d_Ydct));
  CUDA_CHECK(cudaFree(f->residuals->d_Udct));
  CUDA_CHECK(cudaFree(f->residuals->d_Vdct));

  CUDA_CHECK(cudaFree(f->d_mbs[Y_COMPONENT]));
  CUDA_CHECK(cudaFree(f->d_mbs[U_COMPONENT]));
  CUDA_CHECK(cudaFree(f->d_mbs[V_COMPONENT]));


  // cpu
  free(f->recons->Y);
  free(f->recons->U);
  free(f->recons->V);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->Y);
  free(f->predicted->U);
  free(f->predicted->V);
  free(f->predicted);

  free(f->mbs[Y_COMPONENT]);
  free(f->mbs[U_COMPONENT]);
  free(f->mbs[V_COMPONENT]);

  free(f);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

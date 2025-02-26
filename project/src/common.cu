#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t num_blocks_luma, size_t num_blocks_chroma)
{
  struct c63_pipeline *pipe = (c63_pipeline*) calloc(1, sizeof(struct c63_pipeline));
  if (pipe == NULL) { return NULL; }

  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_orig_Y, frame_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_orig_U, chroma_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_orig_V, chroma_size));

  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_recons_Y, frame_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_recons_U, chroma_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_recons_V, chroma_size));

  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_refframe_Y, frame_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_refframe_U, chroma_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_refframe_V, chroma_size));

  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_predicted_Y, frame_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_predicted_U, chroma_size));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_predicted_V, chroma_size));

  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_mbs[Y_COMPONENT], num_blocks_luma * sizeof(struct macroblock)));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_mbs[U_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));
  CUDA_ASSERT(cudaMalloc((void**)&pipe->d_mbs[V_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));

  return pipe;
}

void c63_pipeline_free(struct c63_pipeline *pipe)
{
  CUDA_ASSERT(cudaFree(pipe->d_orig_Y));
  CUDA_ASSERT(cudaFree(pipe->d_orig_U));
  CUDA_ASSERT(cudaFree(pipe->d_orig_V));

  CUDA_ASSERT(cudaFree(pipe->d_recons_Y));
  CUDA_ASSERT(cudaFree(pipe->d_recons_U));
  CUDA_ASSERT(cudaFree(pipe->d_recons_V));

  CUDA_ASSERT(cudaFree(pipe->d_refframe_Y));
  CUDA_ASSERT(cudaFree(pipe->d_refframe_U));
  CUDA_ASSERT(cudaFree(pipe->d_refframe_V));

  CUDA_ASSERT(cudaFree(pipe->d_predicted_Y));
  CUDA_ASSERT(cudaFree(pipe->d_predicted_U));
  CUDA_ASSERT(cudaFree(pipe->d_predicted_V));

  CUDA_ASSERT(cudaFree(pipe->d_mbs[Y_COMPONENT]));
  CUDA_ASSERT(cudaFree(pipe->d_mbs[U_COMPONENT]));
  CUDA_ASSERT(cudaFree(pipe->d_mbs[V_COMPONENT]));
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  frame *f = (frame*)malloc(sizeof(struct frame));

  f->orig = image;

  f->recons = (yuv_t*)malloc(sizeof(yuv_t));
  f->recons->Y = (uint8_t*)malloc(cm->luma_size);
  f->recons->U = (uint8_t*)malloc(cm->chroma_size);
  f->recons->V = (uint8_t*)malloc(cm->chroma_size);

  f->predicted = (yuv_t*)malloc(sizeof(yuv_t));
  f->predicted->Y = (uint8_t*)calloc(cm->luma_size, sizeof(uint8_t));
  f->predicted->U = (uint8_t*)calloc(cm->chroma_size, sizeof(uint8_t));
  f->predicted->V = (uint8_t*)calloc(cm->chroma_size, sizeof(uint8_t));

  f->residuals = (dct_t*)malloc(sizeof(dct_t));
  f->residuals->Ydct = (int16_t*)calloc(cm->luma_size, sizeof(int16_t));
  f->residuals->Udct = (int16_t*)calloc(cm->chroma_size, sizeof(int16_t));
  f->residuals->Vdct = (int16_t*)calloc(cm->chroma_size, sizeof(int16_t));

  f->mbs[Y_COMPONENT] = (macroblock*)calloc(cm->num_mbs_luma, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] = (macroblock*)calloc(cm->num_mbs_chroma, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] = (macroblock*)calloc(cm->num_mbs_chroma, sizeof(struct macroblock));

  return f;
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

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


struct frame* prepare_next_frame(struct c63_common *cm, yuv_t *image)
{
  // move old out of the way
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;

  return create_frame(cm, image);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

int fpeek(FILE *stream)
{
  int c;
  c = fgetc(stream);
  ungetc(c, stream);
  return c;
}


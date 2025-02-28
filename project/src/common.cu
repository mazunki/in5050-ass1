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

  CUDA_ASSERT(cudaMalloc(&pipe->d_orig_Y, frame_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_orig_U, chroma_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_orig_V, chroma_size));

  CUDA_ASSERT(cudaMalloc(&pipe->d_recons_Y, frame_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_recons_U, chroma_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_recons_V, chroma_size));

  CUDA_ASSERT(cudaMalloc(&pipe->d_refframe_Y, frame_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_refframe_U, chroma_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_refframe_V, chroma_size));

  CUDA_ASSERT(cudaMalloc(&pipe->d_predicted_Y, frame_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_predicted_U, chroma_size));
  CUDA_ASSERT(cudaMalloc(&pipe->d_predicted_V, chroma_size));

  CUDA_ASSERT(cudaMalloc(&pipe->d_mbs[Y_COMPONENT], num_blocks_luma * sizeof(struct macroblock)));
  CUDA_ASSERT(cudaMalloc(&pipe->d_mbs[U_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));
  CUDA_ASSERT(cudaMalloc(&pipe->d_mbs[V_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));

  CUDA_ASSERT(cudaHostAlloc(&pipe->h_refframe, sizeof(yuv_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_refframe->Y, frame_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_refframe->U, chroma_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_refframe->V, chroma_size, cudaHostAllocDefault));

  CUDA_ASSERT(cudaHostAlloc(&pipe->h_recons, sizeof(yuv_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_recons->Y, frame_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_recons->U, chroma_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_recons->V, chroma_size, cudaHostAllocDefault));

  CUDA_ASSERT(cudaHostAlloc(&pipe->h_predicted, sizeof(yuv_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_predicted->Y, frame_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_predicted->U, chroma_size, cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_predicted->V, chroma_size, cudaHostAllocDefault));

  CUDA_ASSERT(cudaHostAlloc(&pipe->h_residuals, sizeof(dct_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_residuals->Ydct, frame_size * sizeof(int16_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_residuals->Udct, chroma_size * sizeof(int16_t), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_residuals->Vdct, chroma_size * sizeof(int16_t), cudaHostAllocDefault));

  CUDA_ASSERT(cudaHostAlloc(&pipe->h_mbs[Y_COMPONENT], num_blocks_luma * sizeof(macroblock), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_mbs[U_COMPONENT], num_blocks_chroma * sizeof(macroblock), cudaHostAllocDefault));
  CUDA_ASSERT(cudaHostAlloc(&pipe->h_mbs[V_COMPONENT], num_blocks_chroma * sizeof(macroblock), cudaHostAllocDefault));

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_estimate_V));

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_Y));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_U));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_compensate_V));

  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_macroblocks));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_predictions));
  CUDA_ASSERT(cudaStreamCreate(&pipe->stream_image));

  return pipe;
}

void c63_pipeline_free(struct c63_pipeline *pipe)
{
  // device memory
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

  CUDA_ASSERT(cudaFreeHost(pipe->h_recons->Y));
  CUDA_ASSERT(cudaFreeHost(pipe->h_recons->U));
  CUDA_ASSERT(cudaFreeHost(pipe->h_recons->V));

  CUDA_ASSERT(cudaFreeHost(pipe->h_refframe->Y));
  CUDA_ASSERT(cudaFreeHost(pipe->h_refframe->U));
  CUDA_ASSERT(cudaFreeHost(pipe->h_refframe->V));

  CUDA_ASSERT(cudaFreeHost(pipe->h_predicted->Y));
  CUDA_ASSERT(cudaFreeHost(pipe->h_predicted->U));
  CUDA_ASSERT(cudaFreeHost(pipe->h_predicted->V));

  CUDA_ASSERT(cudaFreeHost(pipe->h_residuals->Ydct));
  CUDA_ASSERT(cudaFreeHost(pipe->h_residuals->Udct));
  CUDA_ASSERT(cudaFreeHost(pipe->h_residuals->Vdct));

  CUDA_ASSERT(cudaFreeHost(pipe->h_mbs[Y_COMPONENT]));
  CUDA_ASSERT(cudaFreeHost(pipe->h_mbs[U_COMPONENT]));
  CUDA_ASSERT(cudaFreeHost(pipe->h_mbs[V_COMPONENT]));

  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_estimate_Y));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_estimate_U));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_estimate_V));

  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_compensate_Y));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_compensate_U));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_compensate_V));

  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_macroblocks));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_predictions));
  CUDA_ASSERT(cudaStreamDestroy(pipe->stream_image));
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  frame *f = (frame*)malloc(sizeof(struct frame));

  f->orig = image;

  f->recons = cm->pipe->h_recons;
  f->predicted = cm->pipe->h_predicted;
  f->residuals = cm->pipe->h_residuals;

  f->mbs[Y_COMPONENT] = cm->pipe->h_mbs[Y_COMPONENT];
  f->mbs[U_COMPONENT] = cm->pipe->h_mbs[U_COMPONENT];
  f->mbs[V_COMPONENT] = cm->pipe->h_mbs[V_COMPONENT];

  return f;
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (f == NULL) { return; }
  free(f->orig);
  free(f);
}

struct frame* prepare_next_frame(struct c63_common *cm)
{
  // move old out of the way
  cm->refframe = cm->curframe;

  // new frame
  frame *f = (frame*)malloc(sizeof(struct frame));
  if (f == NULL) { return NULL; }

  f->orig = cm->frame_buffer[cm->fb_curr_index];

  if (cm->frames_since_keyframe != 0)
  {
    yuv_t *temp = cm->pipe->h_refframe;
    cm->pipe->h_refframe = cm->pipe->h_recons;
    cm->pipe->h_recons = temp;

    SWAP_POINTERS(cm->pipe->h_refframe, cm->pipe->h_recons, yuv_t *);
    SWAP_POINTERS(cm->pipe->d_refframe_Y, cm->pipe->d_recons_Y, uint8_t *);
    SWAP_POINTERS(cm->pipe->d_refframe_U, cm->pipe->d_recons_U, uint8_t *);
    SWAP_POINTERS(cm->pipe->d_refframe_V, cm->pipe->d_recons_V, uint8_t *);
  }

  f->recons = cm->pipe->h_recons;
  f->predicted = cm->pipe->h_predicted;
  f->residuals = cm->pipe->h_residuals;

  f->mbs[Y_COMPONENT] = cm->pipe->h_mbs[Y_COMPONENT];
  f->mbs[U_COMPONENT] = cm->pipe->h_mbs[U_COMPONENT];
  f->mbs[V_COMPONENT] = cm->pipe->h_mbs[V_COMPONENT];

  return f;
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


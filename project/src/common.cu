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

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t macroblock_count)
{
  struct c63_pipeline *pipe = (c63_pipeline*) calloc(1, sizeof(struct c63_pipeline));
  if (pipe == NULL) { return NULL; }

  // streams
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_memcpy));
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_compute));

  pipe->h_recons = (yuv_t*)calloc(1, sizeof(yuv_t));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_recons->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_recons->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_recons->V, chroma_size, cudaHostAllocMapped));

  pipe->h_predicted = (yuv_t*)calloc(1, sizeof(yuv_t));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_predicted->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_predicted->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_predicted->V, chroma_size, cudaHostAllocMapped));

  pipe->h_residuals = (dct_t*)calloc(1, sizeof(dct_t));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_residuals->Ydct, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_residuals->Udct, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->h_residuals->Vdct, chroma_size, cudaHostAllocMapped));


  // pinned cpu
  pipe->input = (struct c63_input*)calloc(1, sizeof(struct c63_input));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig_Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig_U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig_V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe_Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe_U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe_V, chroma_size, cudaHostAllocMapped));


  pipe->output = (struct c63_output*)calloc(1, sizeof(struct c63_output));

  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_predicted_Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_predicted_U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_predicted_V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_residuals_Y, frame_size * sizeof(int16_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_residuals_U, chroma_size * sizeof(int16_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void **)&pipe->output->h_residuals_V, chroma_size * sizeof(int16_t), cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_mbs[Y_COMPONENT], macroblock_count*sizeof(struct macroblock), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_mbs[U_COMPONENT], macroblock_count*sizeof(struct macroblock), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_mbs[V_COMPONENT], macroblock_count*sizeof(struct macroblock), cudaHostAllocMapped));

  // gpu
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void **)&pipe->d_predicted_Y, frame_size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_predicted_U, chroma_size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_predicted_V, chroma_size * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc((void **)&pipe->d_recons_Y, frame_size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_recons_U, chroma_size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_recons_V, chroma_size * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc((void **)&pipe->d_residuals_Y, frame_size * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_residuals_U, chroma_size * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc((void **)&pipe->d_residuals_V, chroma_size * sizeof(int16_t)));


  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[Y_COMPONENT], macroblock_count*sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[U_COMPONENT], macroblock_count*sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[V_COMPONENT], macroblock_count*sizeof(struct macroblock)));


  return pipe;
}

void c63_pipeline_free(struct c63_pipeline *pipe)
{
  if (pipe == NULL) { return; }

  // cpu
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig_Y));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig_U));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig_V));

  CUDA_CHECK(cudaFreeHost(pipe->input->h_refframe_Y));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_refframe_U));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_refframe_V));

  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals_Y));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals_U));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals_V));

  CUDA_CHECK(cudaFreeHost(pipe->output->h_mbs[Y_COMPONENT]));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_mbs[U_COMPONENT]));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_mbs[V_COMPONENT]));

  free(pipe->input);
  free(pipe->output);


  // gpu
  CUDA_CHECK(cudaFree(pipe->d_orig_Y));
  CUDA_CHECK(cudaFree(pipe->d_orig_U));
  CUDA_CHECK(cudaFree(pipe->d_orig_V));

  CUDA_CHECK(cudaFree(pipe->d_refframe_Y));
  CUDA_CHECK(cudaFree(pipe->d_refframe_U));
  CUDA_CHECK(cudaFree(pipe->d_refframe_V));

  CUDA_CHECK(cudaFree(pipe->d_residuals_Y));
  CUDA_CHECK(cudaFree(pipe->d_residuals_U));
  CUDA_CHECK(cudaFree(pipe->d_residuals_V));

  CUDA_CHECK(cudaFree(pipe->d_recons_Y));
  CUDA_CHECK(cudaFree(pipe->d_recons_U));
  CUDA_CHECK(cudaFree(pipe->d_recons_V));

  CUDA_CHECK(cudaFree(pipe->d_mbs[Y_COMPONENT]));
  CUDA_CHECK(cudaFree(pipe->d_mbs[U_COMPONENT]));
  CUDA_CHECK(cudaFree(pipe->d_mbs[V_COMPONENT]));

  // streams
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_memcpy));
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_compute));

  free(pipe);
}

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


struct frame* create_frame_cuda(struct c63_common *cm)
{
  frame *f = (frame*)malloc(sizeof(struct frame));
  if (f == NULL) { return NULL; }

  f->orig = (yuv_t *)malloc(sizeof(yuv_t));
  if (f->orig == NULL) {
    free(f);
    return NULL;
  }

  // cpu
  f->orig->Y = cm->pipe->input->h_orig_Y;
  f->orig->U = cm->pipe->input->h_orig_U;
  f->orig->V = cm->pipe->input->h_orig_V;

  f->mbs[Y_COMPONENT] = cm->pipe->output->h_mbs[Y_COMPONENT];
  f->mbs[U_COMPONENT] = cm->pipe->output->h_mbs[U_COMPONENT];
  f->mbs[V_COMPONENT] = cm->pipe->output->h_mbs[V_COMPONENT];

  f->recons = cm->pipe->h_recons;
  f->predicted = cm->pipe->h_predicted;
  f->residuals = cm->pipe->h_residuals;

  f->predicted->Y = cm->pipe->h_predicted->Y;
  f->predicted->U = cm->pipe->h_predicted->U;
  f->predicted->V = cm->pipe->h_predicted->V;

  f->recons->Y = cm->pipe->h_recons->Y;
  f->recons->U = cm->pipe->h_recons->U;
  f->recons->V = cm->pipe->h_recons->V;

  f->residuals->Ydct = cm->pipe->h_residuals->Ydct;
  f->residuals->Udct = cm->pipe->h_residuals->Udct;
  f->residuals->Vdct = cm->pipe->h_residuals->Vdct;

  // gpu
  f->d_mbs[Y_COMPONENT] = cm->pipe->d_mbs[Y_COMPONENT];
  f->d_mbs[U_COMPONENT] = cm->pipe->d_mbs[U_COMPONENT];
  f->d_mbs[V_COMPONENT] = cm->pipe->d_mbs[V_COMPONENT];

  return f;
}

void destroy_frame_cuda(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (f == NULL) { return; }

  free(f->orig);
  free(f);
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

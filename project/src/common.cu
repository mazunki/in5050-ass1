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

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t num_blocks_luma, size_t num_blocks_chroma)
{
  struct c63_pipeline *pipe = (c63_pipeline*) calloc(1, sizeof(struct c63_pipeline));
  if (pipe == NULL) { return NULL; }

  // streams
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_estimate));
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_compensate));
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_transfer_input));
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_transfer_macroblocks));
  CUDA_CHECK(cudaStreamCreate(&pipe->stream_transfer_predictions));

  // pinned cpu
  pipe->input = (struct c63_input*)calloc(1, sizeof(struct c63_input));
  pipe->output = (struct c63_output*)calloc(1, sizeof(struct c63_output));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig, sizeof(yuv_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_orig->V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe, sizeof(yuv_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->input->h_refframe->V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_recons, sizeof(yuv_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_recons->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_recons->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_recons->V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_predicted, sizeof(yuv_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_predicted->Y, frame_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_predicted->U, chroma_size, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_predicted->V, chroma_size, cudaHostAllocMapped));

  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_residuals, sizeof(dct_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_residuals->Ydct, frame_size * sizeof(int16_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_residuals->Udct, chroma_size * sizeof(int16_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc((void**)&pipe->output->h_residuals->Vdct, chroma_size * sizeof(int16_t), cudaHostAllocMapped));

  // gpu memory
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_orig_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void**)&pipe->d_recons_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_recons_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_recons_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_refframe_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void**)&pipe->d_predicted_Y, frame_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_predicted_U, chroma_size));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_predicted_V, chroma_size));

  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[Y_COMPONENT], num_blocks_luma * sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[U_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));
  CUDA_CHECK(cudaMalloc((void**)&pipe->d_mbs[V_COMPONENT], num_blocks_chroma * sizeof(struct macroblock)));

  return pipe;
}

void c63_pipeline_free(struct c63_pipeline *pipe)
{
  if (pipe == NULL) { return; }

  // cpu
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig->Y));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig->U));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig->V));
  CUDA_CHECK(cudaFreeHost(pipe->input->h_orig));

  CUDA_CHECK(cudaFreeHost(pipe->output->h_recons->Y));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_recons->U));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_recons->V));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_recons));

  CUDA_CHECK(cudaFreeHost(pipe->output->h_predicted->Y));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_predicted->U));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_predicted->V));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_predicted));

  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals->Ydct));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals->Udct));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals->Vdct));
  CUDA_CHECK(cudaFreeHost(pipe->output->h_residuals));

  free(pipe->input);
  free(pipe->output);

  // gpu
  CUDA_CHECK(cudaFree(pipe->d_orig_Y));
  CUDA_CHECK(cudaFree(pipe->d_orig_U));
  CUDA_CHECK(cudaFree(pipe->d_orig_V));

  CUDA_CHECK(cudaFree(pipe->d_recons_Y));
  CUDA_CHECK(cudaFree(pipe->d_recons_U));
  CUDA_CHECK(cudaFree(pipe->d_recons_V));

  CUDA_CHECK(cudaFree(pipe->d_refframe_Y));
  CUDA_CHECK(cudaFree(pipe->d_refframe_U));
  CUDA_CHECK(cudaFree(pipe->d_refframe_V));


  CUDA_CHECK(cudaFree(pipe->d_mbs[Y_COMPONENT]));
  CUDA_CHECK(cudaFree(pipe->d_mbs[U_COMPONENT]));
  CUDA_CHECK(cudaFree(pipe->d_mbs[V_COMPONENT]));

  // streams
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_estimate));
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_compensate));
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_transfer_input));
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_transfer_macroblocks));
  CUDA_CHECK(cudaStreamDestroy(pipe->stream_transfer_predictions));

  free(pipe);
}

struct frame* prepare_next_frame(struct c63_common *cm)
{
  c63_pipeline *pipe = cm->pipe;

  // move old out of the way
  destroy_frame_cuda(cm->refframe);
  cm->refframe = cm->curframe;

  // new frame
  frame *f = (frame*)malloc(sizeof(struct frame));
  if (f == NULL) { return NULL; }

  f->orig = cm->pipe->input->h_orig;

  CUDA_CHECK(cudaMemcpyAsync(pipe->d_orig_Y, f->orig->Y, cm->frame_size, cudaMemcpyHostToDevice, pipe->stream_transfer_input));
  CUDA_CHECK(cudaMemcpyAsync(pipe->d_orig_U, f->orig->U, cm->chroma_size, cudaMemcpyHostToDevice, pipe->stream_transfer_input));
  CUDA_CHECK(cudaMemcpyAsync(pipe->d_orig_V, f->orig->V, cm->chroma_size, cudaMemcpyHostToDevice, pipe->stream_transfer_input));

  if (cm->frames_since_keyframe != 0)
  {
    yuv_t *temp = cm->pipe->input->h_refframe;
    cm->pipe->input->h_refframe = cm->pipe->output->h_recons;
    cm->pipe->output->h_recons = temp;

    cm->pipe->d_refframe_Y = cm->pipe->d_recons_Y;
    cm->pipe->d_refframe_U = cm->pipe->d_recons_U;
    cm->pipe->d_refframe_V = cm->pipe->d_recons_V;
  }

  f->recons = cm->pipe->output->h_recons;
  f->predicted = cm->pipe->output->h_predicted;
  f->residuals = cm->pipe->output->h_residuals;

  f->mbs[Y_COMPONENT] = (macroblock*)calloc(cm->num_blocks_luma, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] = (macroblock*)calloc(cm->num_blocks_chroma, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] = (macroblock*)calloc(cm->num_blocks_chroma, sizeof(struct macroblock));

  return f;
}

void destroy_frame_cuda(struct frame *f)
{
  if (f == NULL) { return; }
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


// used by the decoder
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



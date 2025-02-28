#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file, struct c63_common *cm, int fb_index)
{
  size_t len = 0;

  yuv_t *image = cm->frame_buffer[fb_index];

  uint8_t *Y = image->Y;
  uint8_t *U = image->U;
  uint8_t *V = image->V;

  /* Read Y. The size of Y is the same as the size of the image. The indices
     represents the color component (0 is Y, 1 is U, and 2 is V) */
  len += fread(Y, 1, cm->width*cm->height, file);

  /* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
     because (height/2)*(width/2) = (height*width)/4. */
  len += fread(U, 1, (cm->width*cm->height)/4, file);

  /* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
  len += fread(V, 1, (cm->width*cm->height)/4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  else if (len != cm->width*cm->height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);

    free(image->Y);
    free(image->U);
    free(image->V);
    free(image);

    return NULL;
  }

  return image;
}

static void c63_encode_image(struct c63_common *cm)
{
  cm->curframe = prepare_next_frame(cm);

  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;
  }
  else {
    cm->curframe->keyframe = 0;
  }

  if (!cm->curframe->keyframe)
  {
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_Y, cm->refframe->recons->Y, cm->luma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_U, cm->refframe->recons->U, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_V, cm->refframe->recons->V, cm->chroma_size, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_orig_Y, cm->curframe->orig->Y, cm->luma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_orig_U, cm->curframe->orig->U, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_orig_V, cm->curframe->orig->V, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaDeviceSynchronize());

    /** Motion Estimation
     *   @param[in]  d_orig
     *   @param[in]  d_ref
     *   @param[out] d_mbs
     */
    c63_motion_estimate(cm);
    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], cm->pipe->d_mbs[Y_COMPONENT], cm->num_mbs_luma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(cm->curframe->mbs[U_COMPONENT], cm->pipe->d_mbs[U_COMPONENT], cm->num_mbs_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(cm->curframe->mbs[V_COMPONENT], cm->pipe->d_mbs[V_COMPONENT], cm->num_mbs_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_Y, cm->refframe->recons->Y, cm->luma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_U, cm->refframe->recons->U, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(cm->pipe->d_refframe_V, cm->refframe->recons->V, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaDeviceSynchronize());

    /** Motion Compensation (gpu function)
     *   @param[in]  d_mbs
     *   @param[out] d_predicted
     *   @param[in]  d_ref
     */
    c63_motion_compensate(cm);
    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(cm->curframe->predicted->Y, cm->pipe->d_predicted_Y, cm->luma_size, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(cm->curframe->predicted->U, cm->pipe->d_predicted_U, cm->chroma_size, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(cm->curframe->predicted->V, cm->pipe->d_predicted_V, cm->chroma_size, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  /** quantize (slow CPU-only function)
   *   @param[in]  orig
   *   @param[in]  predicted
   *   @param[out] residuals
   */
  dct_quantize(cm->curframe->orig->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
  dct_quantize(cm->curframe->orig->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->curframe->residuals->Udct, cm->quanttbl[U_COMPONENT]);
  dct_quantize(cm->curframe->orig->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct, cm->quanttbl[V_COMPONENT]);

  /* dequantize (slow CPU-only function)
          @param[in]  residuals
          @param[in]  predicted
          @param[out] recons
  */
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  /** save buffer (slow write-to-disk function)
   *   @param[in]  cm->curframe->residuals->{Y,U,V}dct
   *   @param[in]  mb->curframe->mbs
   *   @param[out] cm->e_ctx.fp (write to disk)
   */
  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  c63_common *cm = (c63_common*)calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->mb_cols_luma = cm->ypw / MACROBLOCK_SIZE;
  cm->mb_rows_luma = cm->yph / MACROBLOCK_SIZE;
  cm->mb_cols_chroma = cm->upw / MACROBLOCK_SIZE;
  cm->mb_rows_chroma = cm->uph / MACROBLOCK_SIZE;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  cm->luma_size = cm->ypw * cm->yph;
  cm->chroma_size = cm->upw * cm->uph;
  cm->num_mbs_luma = cm->mb_rows_luma * cm->mb_cols_luma;
  cm->num_mbs_chroma = cm->mb_rows_chroma * cm->mb_cols_chroma;

  cm->pipe = c63_pipeline_init(cm->luma_size, cm->chroma_size, cm->num_mbs_luma, cm->num_mbs_chroma);

  for (int i=0; i < FRAMEBUFFER_SIZE; i++) {
    cm->frame_buffer[i] = (yuv_t *)malloc(sizeof(yuv_t));
    CUDA_ASSERT(cudaHostAlloc(&cm->frame_buffer[i]->Y, cm->luma_size, cudaHostAllocDefault));
    CUDA_ASSERT(cudaHostAlloc(&cm->frame_buffer[i]->U, cm->chroma_size, cudaHostAllocDefault));
    CUDA_ASSERT(cudaHostAlloc(&cm->frame_buffer[i]->V, cm->chroma_size, cudaHostAllocDefault));
  }
  cm->fb_curr_index = 0;

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  CUDA_ASSERT(cudaFreeHost(cm->curframe->orig->Y));
  CUDA_ASSERT(cudaFreeHost(cm->curframe->orig->U));
  CUDA_ASSERT(cudaFreeHost(cm->curframe->orig->V));
  free(cm->curframe->orig);

  c63_pipeline_free(cm->pipe);
  destroy_frame(cm->curframe);
  free(cm);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes) { fprintf(stderr, "Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;
  if (read_yuv(infile, cm, cm->fb_curr_index) == NULL) {
    exit(EXIT_FAILURE);
  }

  do {
    int fb_next = (cm->fb_curr_index+1) % FRAMEBUFFER_SIZE;

    if (fpeek(infile) != EOF) {
      if (read_yuv(infile, cm, fb_next) == NULL) {
        exit(EXIT_FAILURE);
      }
    } else {
      cm->frame_buffer[fb_next] = NULL;
    }

    printf("Encoding frame %d...", numframes+1);
    c63_encode_image(cm);

    printf(" done!\n");

    cm->fb_curr_index = fb_next;
    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  } while (cm->frame_buffer[cm->fb_curr_index] != NULL);


  printf("Completed encoding! Encoded %d frames\n", numframes);



  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}


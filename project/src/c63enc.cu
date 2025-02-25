#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "c63.h"
#include "c63_write.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

/* getopt */
extern int optind;
extern char *optarg;

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv(FILE *file, struct c63_common *cm)
{
  size_t len = 0;

  uint8_t *Y = cm->pipe->input->h_orig->Y;
  uint8_t *U = cm->pipe->input->h_orig->U;
  uint8_t *V = cm->pipe->input->h_orig->V;

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

  if (feof(file))
  {
    fprintf(stderr, "i really love potatoes and lise made me not delete this line she is holding me hostage\n");
    return -1;
  }
  else if (len != cm->width*cm->height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", cm->height, cm->width);

    return -2;
  }

  return 0;
}

static void c63_encode_image(struct c63_common *cm)
{
  c63_pipeline *pipe = cm->pipe;

  DEBUG("frame start");
  cm->curframe = prepare_next_frame(cm);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    memset(cm->pipe->output->h_predicted->Y, 0, cm->frame_size);
    memset(cm->pipe->output->h_predicted->U, 0, cm->chroma_size);
    memset(cm->pipe->output->h_predicted->V, 0, cm->chroma_size);

    memset(cm->pipe->output->h_residuals->Ydct, 0, cm->frame_size * sizeof(int16_t));
    memset(cm->pipe->output->h_residuals->Udct, 0, cm->chroma_size * sizeof(int16_t));
    memset(cm->pipe->output->h_residuals->Vdct, 0, cm->chroma_size * sizeof(int16_t));


    fprintf(stderr, " (keyframe) ");
  }
  else { cm->curframe->keyframe = 0; }

  if (!cm->curframe->keyframe)
  {
    /* Motion Estimation */
    CUDA_CHECK(cudaMemcpy(pipe->d_orig_Y, pipe->input->h_orig->Y, cm->frame_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pipe->d_orig_U, pipe->input->h_orig->U, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pipe->d_orig_V, pipe->input->h_orig->V, cm->chroma_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(pipe->d_refframe_Y, pipe->input->h_refframe->Y, cm->frame_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pipe->d_refframe_U, pipe->input->h_refframe->U, cm->chroma_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pipe->d_refframe_V, pipe->input->h_refframe->V, cm->chroma_size, cudaMemcpyHostToDevice));

    c63_motion_estimate(cm);
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], pipe->d_mbs[Y_COMPONENT], cm->num_blocks_luma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[U_COMPONENT], pipe->d_mbs[U_COMPONENT], cm->num_blocks_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cm->curframe->mbs[V_COMPONENT], pipe->d_mbs[V_COMPONENT], cm->num_blocks_chroma * sizeof(struct macroblock), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Motion Compensation */
    c63_motion_compensate_cuda(cm);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  dct_quantize(cm->pipe->input->h_orig->Y, cm->pipe->output->h_predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT], cm->pipe->output->h_residuals->Ydct, cm->quanttbl[Y_COMPONENT]);
  dct_quantize(cm->pipe->input->h_orig->U, cm->pipe->output->h_predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], cm->pipe->output->h_residuals->Udct, cm->quanttbl[U_COMPONENT]);
  dct_quantize(cm->pipe->input->h_orig->V, cm->pipe->output->h_predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], cm->pipe->output->h_residuals->Vdct, cm->quanttbl[V_COMPONENT]);

  dequantize_idct(cm->pipe->output->h_residuals->Ydct, cm->pipe->output->h_predicted->Y, cm->ypw, cm->yph, cm->pipe->output->h_recons->Y, cm->quanttbl[Y_COMPONENT]);
  dequantize_idct(cm->pipe->output->h_residuals->Udct, cm->pipe->output->h_predicted->U, cm->upw, cm->uph, cm->pipe->output->h_recons->U, cm->quanttbl[U_COMPONENT]);
  dequantize_idct(cm->pipe->output->h_residuals->Vdct, cm->pipe->output->h_predicted->V, cm->vpw, cm->vph, cm->pipe->output->h_recons->V, cm->quanttbl[V_COMPONENT]);

  CUDA_CHECK(cudaDeviceSynchronize());

  DEBUG("c63enc\n");
  for (int i=0; i<10; i++) {
    DEBUG("frame %d", cm->framenum);
    DEBUG("MV Y[%d]: (%d, %d)", i, cm->curframe->mbs[Y_COMPONENT][i].mv_x, cm->curframe->mbs[Y_COMPONENT][i].mv_y);
    DEBUG("MV U[%d]: (%d, %d)", i, cm->curframe->mbs[U_COMPONENT][i].mv_x, cm->curframe->mbs[U_COMPONENT][i].mv_y);
    DEBUG("MV V[%d]: (%d, %d)", i, cm->curframe->mbs[V_COMPONENT][i].mv_x, cm->curframe->mbs[V_COMPONENT][i].mv_y);
    DEBUG("predicted [%d]: (%d, %d, %d)", i, cm->curframe->predicted->Y[i], cm->curframe->predicted->U[i], cm->curframe->predicted->V[i]);
    DEBUG("recons [%d]: (%d, %d, %d)", i, cm->curframe->recons->Y[i], cm->curframe->recons->U[i], cm->curframe->recons->V[i]);
    DEBUG("residuals [%d]: (%d, %d, %d)", i, cm->curframe->residuals->Ydct[i], cm->curframe->residuals->Udct[i], cm->curframe->residuals->Vdct[i]);
    DEBUG("h_refframe [%d]: (%d, %d, %d)", i, cm->pipe->input->h_refframe->Y[i], cm->pipe->input->h_refframe->U[i], cm->pipe->input->h_refframe->V[i]);
  }
  

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;

  DEBUG("frame complete");
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

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

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

  cm->frame_size = cm->ypw * cm->yph;
  cm->chroma_size = cm->upw * cm->uph;
  cm->num_blocks_luma = cm->mb_rows * cm->mb_cols;
  cm->num_blocks_chroma = (cm->mb_rows / 2) * (cm->mb_cols / 2);
  cm->macroblock_count = cm->num_blocks_luma + 2 * cm->num_blocks_chroma;

  cm->pipe = c63_pipeline_init(cm->frame_size, cm->chroma_size, cm->num_blocks_luma, cm->num_blocks_chroma);

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  if (cm == NULL) { return; }

  destroy_frame_cuda(cm->curframe);
  c63_pipeline_free(cm->pipe);

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
  int width, height;

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

  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  while (1)
  {
    int status = read_yuv(infile, cm);
    if (status < 0) {
      break;
    }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm);

    printf("Done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  //int i, j;
  //for (i = 0; i < 2; ++i)
  //{
  //  printf("int freq[] = {");
  //  for (j = 0; j < ARRAY_SIZE(frequencies[i]); ++j)
  //  {
  //    printf("%d, ", frequencies[i][j]);
  //  }
  //  printf("};\n");
  //}

  return EXIT_SUCCESS;
}

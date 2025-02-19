#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

#include <cuda_runtime.h>
#include "common.h"

#include <cuda_runtime.h>
#include "common.h"

void destroy_frame(frame *f)
{
    if (!f) return;

    // Free CPU memory
    free(f->recons->Y);
    free(f->recons->U);
    free(f->recons->V);
    free(f->recons);

    free(f->predicted->Y);
    free(f->predicted->U);
    free(f->predicted->V);
    free(f->predicted);

    free(f->residuals->Ydct);
    free(f->residuals->Udct);
    free(f->residuals->Vdct);
    free(f->residuals);

    free(f->mbs[Y_COMPONENT]);
    free(f->mbs[U_COMPONENT]);
    free(f->mbs[V_COMPONENT]);

    // Free GPU memory
    cudaFree(f->recons->d_Y);
    cudaFree(f->recons->d_U);
    cudaFree(f->recons->d_V);

    cudaFree(f->predicted->d_Y);
    cudaFree(f->predicted->d_U);
    cudaFree(f->predicted->d_V);

    cudaFree(f->residuals->d_Ydct);
    cudaFree(f->residuals->d_Udct);
    cudaFree(f->residuals->d_Vdct);

    free(f);
}


frame *create_frame(struct c63_common *cm, yuv_t *image)
{
    frame *f = (frame *)malloc(sizeof(frame));
    if (!f) return NULL;

    size_t frame_size = cm->ypw * cm->yph;
    size_t chroma_size = cm->upw * cm->uph;
    size_t num_blocks_Y = cm->mb_rows * cm->mb_cols;
    size_t num_blocks_UV = (cm->mb_rows / 2) * (cm->mb_cols / 2);

    f->orig = image;

    // Allocate CPU memory
    f->recons = (yuv_t *)malloc(sizeof(yuv_t));
    f->recons->Y = (uint8_t *)malloc(frame_size);
    f->recons->U = (uint8_t *)malloc(chroma_size);
    f->recons->V = (uint8_t *)malloc(chroma_size);

    f->predicted = (yuv_t *)malloc(sizeof(yuv_t));
    f->predicted->Y = (uint8_t *)calloc(frame_size, sizeof(uint8_t));
    f->predicted->U = (uint8_t *)calloc(chroma_size, sizeof(uint8_t));
    f->predicted->V = (uint8_t *)calloc(chroma_size, sizeof(uint8_t));

    f->residuals = (dct_t *)malloc(sizeof(dct_t));
    f->residuals->Ydct = (int16_t *)calloc(frame_size, sizeof(int16_t));
    f->residuals->Udct = (int16_t *)calloc(chroma_size, sizeof(int16_t));
    f->residuals->Vdct = (int16_t *)calloc(chroma_size, sizeof(int16_t));

    f->mbs[Y_COMPONENT] = (macroblock *)calloc(num_blocks_Y, sizeof(macroblock));
    f->mbs[U_COMPONENT] = (macroblock *)calloc(num_blocks_UV, sizeof(macroblock));
    f->mbs[V_COMPONENT] = (macroblock *)calloc(num_blocks_UV, sizeof(macroblock));

    // Allocate GPU memory
    cudaMalloc((void **)&f->recons->d_Y, frame_size);
    cudaMalloc((void **)&f->recons->d_U, chroma_size);
    cudaMalloc((void **)&f->recons->d_V, chroma_size);

    cudaMalloc((void **)&f->predicted->d_Y, frame_size);
    cudaMalloc((void **)&f->predicted->d_U, chroma_size);
    cudaMalloc((void **)&f->predicted->d_V, chroma_size);

    cudaMalloc((void **)&f->residuals->d_Ydct, frame_size * sizeof(int16_t));
    cudaMalloc((void **)&f->residuals->d_Udct, chroma_size * sizeof(int16_t));
    cudaMalloc((void **)&f->residuals->d_Vdct, chroma_size * sizeof(int16_t));

    return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp) {
	fwrite(image->Y, 1, w*h, fp);
	fwrite(image->U, 1, w*h/4, fp);
	fwrite(image->V, 1, w*h/4, fp);
}


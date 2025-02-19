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

#define MACROBLOCK_SIZE 8

static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result) {
	int u, v;

	*result = 0;

	for (v = 0; v < MACROBLOCK_SIZE; ++v) {
		for (u = 0; u < MACROBLOCK_SIZE; ++u) {
			*result += abs(block2[v*stride+u] - block1[v*stride+u]);
		}
	}
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct macroblock *mb, int mb_x, int mb_y, uint8_t *orig, uint8_t *ref, int padw, int padh, int range) {
	int left = mb_x * MACROBLOCK_SIZE - range;
	int top = mb_y * MACROBLOCK_SIZE - range;
	int right = mb_x * MACROBLOCK_SIZE + range;
	int bottom = mb_y * MACROBLOCK_SIZE + range;

	/* Make sure we are within bounds of reference frame. TODO: Support partial frame bounds. */
	if (left < 0) { left = 0; }
	if (top < 0) { top = 0; }
	if (right > (padw - MACROBLOCK_SIZE)) { right = padw - MACROBLOCK_SIZE; }
	if (bottom > (padh - MACROBLOCK_SIZE)) { bottom = padh - MACROBLOCK_SIZE; }

	int x, y;

	int mx = mb_x * MACROBLOCK_SIZE;
	int my = mb_y * MACROBLOCK_SIZE;

	int best_sad = INT_MAX;

	for (y = top; y < bottom; ++y) {
		for (x = left; x < right; ++x) {
			int sad;
			sad_block_8x8(orig + (my*padw + mx), ref + (y*padw + x), padw, &sad);

			/* printf("(%4d,%4d) - %d\n", x, y, sad); */

			if (sad < best_sad) {
				mb->mv_x = x - mx;
				mb->mv_y = y - my;
				best_sad = sad;
			}
		}
	}

	/* Here, there should be a threshold on SAD that checks if the motion vector
	 is cheaper than intraprediction. We always assume MV to be beneficial */

	/* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y, best_sad); */

	mb->use_mv = 1;
}

void c63_motion_estimate_kernel(uint8_t *d_orig, uint8_t *d_recons, macroblock *d_mbs, int width, int height, int range) {
    for (int mb_y = 0; mb_y < height / MACROBLOCK_SIZE; ++mb_y) {
        for (int mb_x = 0; mb_x < width / MACROBLOCK_SIZE; ++mb_x) {
            macroblock *mb = &d_mbs[mb_y * (width / MACROBLOCK_SIZE) + mb_x];
            me_block_8x8(mb, mb_x, mb_y, d_orig, d_recons, width, height, range);
        }
    }
}

void c63_motion_estimate(struct c63_common *cm) {
	/* Compare this frame with previous reconstructed frame */
	int width = cm->padw[Y_COMPONENT];
	int height = cm->padh[Y_COMPONENT];
	int range = cm->me_search_range;

	/* Luma */
	c63_motion_estimate_kernel(cm->curframe->orig->Y, cm->refframe->recons->Y, cm->curframe->mbs[Y_COMPONENT], width, height, range);

	/* Chroma */
	range /= 2;  // quarter resolution
	c63_motion_estimate_kernel(cm->curframe->orig->U, cm->refframe->recons->U, cm->curframe->mbs[U_COMPONENT], cm->padw[U_COMPONENT], cm->padh[U_COMPONENT], range);
	c63_motion_estimate_kernel(cm->curframe->orig->V, cm->refframe->recons->V, cm->curframe->mbs[V_COMPONENT], cm->padw[V_COMPONENT], cm->padh[V_COMPONENT], range);
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int color_component) {
	struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/MACROBLOCK_SIZE+mb_x];

	if (!mb->use_mv) { return; }

	int left = mb_x * MACROBLOCK_SIZE;
	int top = mb_y * MACROBLOCK_SIZE;
	int right = left + MACROBLOCK_SIZE;
	int bottom = top + MACROBLOCK_SIZE;

	int w = cm->padw[color_component];

	/* Copy block from ref mandated by MV */
	int x, y;

	for (y = top; y < bottom; ++y) {
		for (x = left; x < right; ++x) {
			predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
		}
	}
}

void c63_motion_compensate(struct c63_common *cm) {
	int mb_x, mb_y;

	/* Luma */
	for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y) {
		for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x) {
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y) {
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x) {
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);
		}
	}
}


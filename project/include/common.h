#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }

// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */

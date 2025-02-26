#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

#define MACROBLOCK_SIZE 8

#define CUDA_ASSERT(call)                                                     \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }

#define CUDA_CHECK()                                                          \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }


#if NDEBUG
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#endif

// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */


#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

#define MACROBLOCK_SIZE 8

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }

#define CUDA_ASSERT()                                                         \
    {                                                                         \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (file %s, line %d)\n",            \
                    cudaGetErrorString(err), __FILE__, __LINE__);             \
            exit(err);                                                        \
        }                                                                     \
    }


#if 0
#define DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG(fmt, ...)
#endif


// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);
void destroy_frame(struct frame *f);

struct frame* prepare_next_frame(struct c63_common *cm);
void destroy_frame_cuda(struct frame *f);

struct c63_pipeline* c63_pipeline_init(size_t frame_size, size_t chroma_size, size_t num_blocks_luma, size_t num_blocks_chroma);
void c63_pipeline_free(struct c63_pipeline *pipe);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

#endif  /* C63_COMMON_H_ */

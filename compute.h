#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#ifdef __cplusplus
extern "C" {
#endif

void compute();
void compute_cuda_init();
void compute_cuda_finalize();

#ifdef __cplusplus
}
#endif

#endif
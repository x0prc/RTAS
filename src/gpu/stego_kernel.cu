#include "config.h"
#include <cuComplex.h>

extern "C" __global__ void embed_kernel(float *cover, float *secret, float *stego, float alpha, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        stego[idx] = cover[idx] + alpha * secret[idx];
    }
}

#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#define checkCudaErrors(call)                                \
    do {                                                     \
        cudaError_t err = call;                              \
        if (err != cudaSuccess) {                            \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while (0)  

float rand_float(float min, float max);


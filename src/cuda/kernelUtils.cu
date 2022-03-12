#include "cuda/kernel.h"
#include <assert.h>
#include <cstdio>
#include <cuda.h>

__global__ void isnanTestKernel(cuCpx *data, int n) { // with grimDim == 1
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (isnan(data[i].x) || isnan(data[i].y)) {
            printf("nan at %d\n", i);
            asm("trap;");
        }
    }
}

__global__ void printVectorKernel(cuCpx *data, int n) { // with gridDim == 1 && blockDim == 1
    for (int i = 0; i < n; i++)
        printf("(%f, %f)", data[i].x, data[i].y);
    printf("\n");
}

__global__ void printVectorParallel(cuCpx* data, int n, int task_id) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && data[tid].x * data[tid].x + data[tid].y * data[tid].y > 0.00001)
        printf("(%d) %x: %lf %lf\n", task_id, tid, data[tid].x, data[tid].y);
}

__global__ void whileTrueKernel() {
    while (true);
}

void isnanTest(cpx* data, int n, cudaStream_t& stream) {
    isnanTestKernel<<<1, 32, 0, stream>>>(reinterpret_cast<cuCpx*>(data), n / 32);
}

void printVector(cuCpx* data, int n, cudaStream_t& stream) {
    // printVectorKernel<<<1, 1, 0, stream>>>(data, n);
    static int task_id = 0;
    task_id ++;
    printVectorParallel<<<(n + 127)/128, 128, 0, stream>>>(data, n, task_id);
}

void printVector(cpx* data, int n, cudaStream_t& stream) {
    printVector(reinterpret_cast<cuCpx*>(data), n, stream);
}

void whileTrue() {
    whileTrueKernel<<<1,1>>>();
}

cuCpx make_cuComplex(value_t x) {
    return make_cuComplex(x, 0.0);
}
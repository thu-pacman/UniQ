#include "entry.h"

#include <cuda_profiler_api.h>
#include "kernel.h"

namespace CudaImpl {

void startProfiler() {
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStart());
    }
}

void stopProfiler() {
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStop());
    }
}

}
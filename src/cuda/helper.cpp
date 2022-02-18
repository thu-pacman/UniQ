#include "cuda/entry.h"
#include <assert.h>
#include "cuda/kernel.h"

namespace CudaImpl {
cpx getAmp (const std::vector<cpx*>& deviceStateVec, int gpuID, idx_t localIdx) {
    checkCudaErrors(cudaSetDevice(gpuID));
    cpx ret;
    cudaMemcpy(&ret, deviceStateVec[gpuID] + localIdx, sizeof(cpx), cudaMemcpyDeviceToHost);
    return ret;
}

void copyBackState(std::vector<cpx>& result, const std::vector<cpx*>& deviceStateVec, int numQubits) {
    result.resize(1ll << numQubits); // very slow ...
#if GPU_BACKEND == 0 || GPU_BACKEND == 2
    cudaMemcpy((cpx*)result.data(), deviceStateVec[0], sizeof(cpx) << numQubits, cudaMemcpyDeviceToHost);
#else
    idx_t elements = 1ll << (numQubits - MyGlobalVars::bit);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaMemcpy((cpx*)result.data() + elements * g, deviceStateVec[g], sizeof(cpx) << (numQubits - MyGlobalVars::bit), cudaMemcpyDeviceToHost);
    }
#endif
}

void destroyState(std::vector<cpx*>& deviceStateVec) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaFree(deviceStateVec[g]);
    }
}

void initGPUMatrix(std::vector<cpx*>& deviceMats, int matQubit, const std::vector<std::unique_ptr<cpx[]>>& matrix) {
    assert(deviceMats.size() == 0);
    deviceMats.clear();
    int n = 1 << matQubit;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        cpx realMat[n][n];
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                realMat[i][j] = matrix[g][i * n + j];
            }
        cpx* mat;
        cudaMalloc(&mat, n * n * sizeof(cpx));
        cudaMemcpyAsync(mat, realMat, n * n * sizeof(cpx), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]);
        deviceMats.push_back(mat);
    }
}


void initCuttPlans(std::vector<cuttHandle*>& cuttPlanPointers, const std::vector<int*>& cuttPermPointers, const std::vector<int>& locals, int numLocalQubits) {
    int total = cuttPlanPointers.size();
    cuttHandle plans[total];
    std::vector<int> dim(numLocalQubits, 2);
    if (total == 0) return;

    checkCudaErrors(cudaSetDevice(0));

    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        checkCuttErrors(cuttPlan(&plans[i], locals[i], dim.data(), cuttPermPointers[i], sizeof(cpx), MyGlobalVars::streams[0], false));
    }

    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        for (int i = 0; i < total; i++) {
            checkCuttErrors(cuttActivatePlan(cuttPlanPointers[i] + g, plans[i], MyGlobalVars::streams[g], g));
        }
    }
}

}
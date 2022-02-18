#pragma once

#include "utils.h"
#include <vector>

typedef unsigned int cuttHandle;

namespace CudaImpl {
// init.cpp
void initCudaObjects();
void initState(std::vector<cpx*> &deviceStateVec, int numQubits);

// profiler.cpp
void startProfiler();
void stopProfiler();

// helper.cpp
cpx getAmp(const std::vector<cpx*>& deviceStateVec, int gpuID, idx_t localIdx);
void copyBackState(std::vector<cpx>& result, const std::vector<cpx*>& deviceStateVec, int numQubits);
void destroyState(std::vector<cpx*>& deviceStateVec);
void initGPUMatrix(std::vector<cpx*>& deviceMats, int matQubit, const std::vector<std::unique_ptr<cpx[]>>& matrix);
void initCuttPlans(std::vector<cuttHandle*>& transPlanPointers, const std::vector<int*>& transPermPointers, const std::vector<int>& locals, int numLocalQubits);
}
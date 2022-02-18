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
cpx getAmp(std::vector<cpx*>& deviceStateVec, int gpuID, idx_t localIdx);
void copyBackState(std::vector<cpx>& result, std::vector<cpx*>& deviceStateVec, int numQubits);
void destroyState(std::vector<cpx*>& deviceStateVec);
void initGPUMatrix(std::vector<cpx*>& deviceMats, int matQubit, std::vector<std::unique_ptr<cpx[]>>& matrix);
void initCuttPlans(std::vector<cuttHandle*>& cuttPlanPointers, std::vector<int*>& cuttPermPointers, std::vector<int>& locals, int numLocalQubits);
}
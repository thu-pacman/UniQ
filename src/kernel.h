#pragma once

#include <vector>
#include <cutt.h>

#include "gate.h"
#include "utils.h"
#include "compiler.h"
#include "circuit.h"

// kernelSimple
void kernelInit(std::vector<cpx*> &deviceStateVec, int numQubits);
void kernelExecSimple(cpx* deviceStateVec, int numQubits, const std::vector<Gate> & gates);
value_t kernelMeasure(cpx* deviceStateVec, int numQubits, int targetQubit);
cpx kernelGetAmp(cpx* deviceStateVec, idx_t idx);
void kernelDeviceToHost(cpx* hostStateVec, cpx* deviceStateVec, int numQubits);
void kernelDestroy(cpx* deviceStateVec);
void cuttPlanInit(std::vector<cuttHandle>& plans);
void packing(int numQubits, const cpx* src, cpx* dest); // result is saved in dest
void unpacking(int numQubits, cpx* src, cpx* buffer); // result is saved in src


// kernelOpt
void initControlIdx();
// call cudaSetDevice() before this function
void copyGatesToSymbol(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID);

// call cudaSetDevice() before this function
void launchExecutor(int gridDim, cpx* deviceStateVec, unsigned int* threadBias, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate, cudaStream_t& stream, int gpuID);


// kernelUtils
void isnanTest(cpx* data, int n, cudaStream_t& stream);
void printVector(cpx* data, int n, cudaStream_t& stream);
void whileTrue();
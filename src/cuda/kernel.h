#pragma once

#include <vector>
#include <cuComplex.h>
#include <cuda.h>
#include <cutt.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#if USE_MPI
#include <mpi.h>
#include <nccl.h>
#endif

#include "gate.h"
#include "utils.h"
#include "compiler.h"
#include "circuit.h"

// kernelSimple
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

static const char *cublasGetErrorString(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "<unknown>";
    }
    UNREACHABLE()
}

static const char *cuttGetErrorString(cuttResult error) {
    switch (error) {
        case CUTT_INVALID_PLAN:
            return "CUTT_INVALID_PLAN";
        case CUTT_INVALID_PARAMETER:
            return "CUTT_INVALID_PARAMETER";
        case CUTT_INVALID_DEVICE:
            return "CUTT_INVALID_DEVICE";
        case CUTT_INTERNAL_ERROR:
            return "CUTT_INTERNAL_ERROR";
        case CUTT_UNDEFINED_ERROR:
            return "CUTT_UNDEFINED_ERROR";
        default:
            return "<unknown>";
    }
    UNREACHABLE()
}

#define checkCudaErrors(stmt) {                                 \
    cudaError_t err = stmt;                            \
    if (err != cudaSuccess) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cudaGetErrorString(err)); \
      exit(1); \
    }                                                  \
}

#define checkCuttErrors(stmt) {                                 \
    cuttResult err = stmt;                            \
    if (err != CUTT_SUCCESS) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cuttGetErrorString(err)); \
      exit(1); \
    }                                                  \
}

#define checkBlasErrors(stmt) { \
    cublasStatus_t err = stmt; \
    if (err != CUBLAS_STATUS_SUCCESS) {                          \
      fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, cublasGetErrorString(err)); \
      exit(1); \
    } \
}

#if USE_MPI
#define checkNCCLErrors(stmt) {                         \
  ncclResult_t err= stmt;                             \
  if (err != ncclSuccess) {                            \
    fprintf(stderr, "%s in file %s, function %s, line %i: %04d %s\n", #stmt, __FILE__, __FUNCTION__, __LINE__, err, ncclGetErrorString(err)); \
      exit(1); \
  }                                                 \
}
#endif

namespace MyGlobalVars {
    extern std::unique_ptr<cudaStream_t[]> streams;
    extern std::unique_ptr<cudaStream_t[]> streams_comm;
    extern std::unique_ptr<cublasHandle_t[]> blasHandles;
    extern std::unique_ptr<cudaEvent_t[]> events;
#if USE_MPI
    extern std::unique_ptr<ncclComm_t[]> ncclComms;
#endif
}

#ifdef USE_DOUBLE
    typedef cuDoubleComplex cuCpx;
    #define make_cuComplex make_cuDoubleComplex
    #define cublasGEMM cublasZgemm
    #define NCCL_FLOAT_TYPE ncclDouble
#else
    typedef cuFloatComplex cuCpx;
    #define make_cuComplex make_cuFloatComplex
    #define cublasGEMM cublasCgemm
    #define NCCL_FLOAT_TYPE ncclFloat
#endif

const int THREAD_DEP = THREAD_DEP_DEFINED; // 1 << THREAD_DEP threads per block
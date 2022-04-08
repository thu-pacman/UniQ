#include "cuda/kernel.h"
#include <cstdio>
#include <assert.h>
#include "logger.h"
using namespace std;

namespace MyGlobalVars {
std::unique_ptr<cudaStream_t[]> streams;
std::unique_ptr<cudaStream_t[]> streams_comm;
std::unique_ptr<cublasHandle_t[]> blasHandles;
std::unique_ptr<cudaEvent_t[]> events;
#if USE_MPI
std::unique_ptr<ncclComm_t[]> ncclComms;
#endif
}

namespace CudaImpl {

void initState(std::vector<cpx*> &deviceStateVec, int numQubits) {
    size_t size = (sizeof(cuCpx) << numQubits) >> MyGlobalVars::bit;
    if ((MyGlobalVars::numGPUs > 1 && !INPLACE) || GPU_BACKEND == 3 || GPU_BACKEND == 4 || MODE == 1) {
        size <<= 1;
    }
#if INPLACE
    size += sizeof(cuCpx) * (1 << (MODE == 2 ? MAX_SLICE * 2 : MAX_SLICE));
#endif
#if GPU_BACKEND == 2
    deviceStateVec.resize(1);
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc(&reinterpret_cast<cuCpx*>(deviceStateVec[0]), sizeof(cuCpx) << numQubits));
    checkCudaErrors(cudaMemsetAsync(reinterpret_cast<cuCpx*>(deviceStateVec[0]), 0, sizeof(cuCpx) << numQubits, MyGlobalVars::streams[0]));
#else
    deviceStateVec.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaMalloc(reinterpret_cast<cuCpx**>(&deviceStateVec[g]), size));
        checkCudaErrors(cudaMemsetAsync(reinterpret_cast<cuCpx*>(deviceStateVec[g]), 0, size, MyGlobalVars::streams[g]));
    }
#endif
    cuCpx one = make_cuComplex(1.0, 0.0);
    if  (!USE_MPI || MyMPI::rank == 0) {
        checkCudaErrors(cudaMemcpyAsync(reinterpret_cast<cuCpx*>(deviceStateVec[0]), &one, sizeof(cuCpx), cudaMemcpyHostToDevice, MyGlobalVars::streams[0])); // state[0] = 1
    }
#if GPU_BACKEND == 1 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
    initControlIdx();
#endif
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
    }
}

void initCudaObjects() {
    checkCudaErrors(cudaGetDeviceCount(&MyGlobalVars::localGPUs));
    #if MODE == 2
        dmAllocGate(MyGlobalVars::localGPUs);
    #endif
    #if ALL_TO_ALL
    if (MyGlobalVars::localGPUs != 1) {
        printf("[error] only support one GPU / rank in ALL_TO_ALL mode");
        UNIMPLEMENTED();
    }
    #endif
    #if USE_MPI
        MyGlobalVars::numGPUs = MyMPI::commSize * MyGlobalVars::localGPUs;
    #else
        MyGlobalVars::numGPUs = MyGlobalVars::localGPUs;
    #endif
    Logger::add("Local GPU: %d", MyGlobalVars::localGPUs);
    MyGlobalVars::bit = get_bit(MyGlobalVars::numGPUs);

    MyGlobalVars::streams = std::make_unique<cudaStream_t[]>(MyGlobalVars::localGPUs);
    MyGlobalVars::streams_comm = std::make_unique<cudaStream_t[]>(MyGlobalVars::localGPUs);
    MyGlobalVars::blasHandles = std::make_unique<cublasHandle_t[]>(MyGlobalVars::localGPUs);
    MyGlobalVars::events = std::make_unique<cudaEvent_t[]>(MyGlobalVars::localGPUs);
    checkCuttErrors(cuttInit());
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        Logger::add("[%d] %s", i, prop.name);
        for (int j = 0; j < MyGlobalVars::localGPUs; j++)
            if (i != j && (i ^ j) < 4) {
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
            }
        checkCudaErrors(cudaStreamCreate(&MyGlobalVars::streams[i]);)
        checkBlasErrors(cublasCreate(&MyGlobalVars::blasHandles[i]));
        checkBlasErrors(cublasSetStream(MyGlobalVars::blasHandles[i], MyGlobalVars::streams[i]));
        checkCudaErrors(cudaStreamCreate(&MyGlobalVars::streams_comm[i]));
        checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        checkCudaErrors(cudaEventCreate(&MyGlobalVars::events[i]));
    }
    #if USE_MPI
        checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
        ncclUniqueId id;
        if (MyMPI::rank == 0)
            checkNCCLErrors(ncclGetUniqueId(&id));
        checkMPIErrors(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
        MyGlobalVars::ncclComms = std::make_unique<ncclComm_t[]>(MyGlobalVars::localGPUs);
        checkNCCLErrors(ncclGroupStart());
        for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
            checkCudaErrors(cudaSetDevice(i));
            checkNCCLErrors(ncclCommInitRank(&MyGlobalVars::ncclComms[i], MyGlobalVars::numGPUs, id, MyMPI::rank * MyGlobalVars::localGPUs + i));
        }
        checkNCCLErrors(ncclGroupEnd());
    #endif
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

}
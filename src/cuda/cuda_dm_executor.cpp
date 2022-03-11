#include "cuda_dm_executor.h"
#include "cuda/kernel.h"

#include <assert.h>

namespace CudaImpl {

CudaDMExecutor::CudaDMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): DMExecutor(deviceStateVec, numQubits, schedule) {
    threadBias.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaMalloc(&threadBias[g], sizeof(idx_t) << THREAD_DEP));
    }
}

void CudaDMExecutor::launchPerGateGroupDM(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    unsigned int blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate, numLocalQubits);
    idx_t gridDim = idx_t(1) << (numLocalQubits - LOCAL_QUBIT_SIZE) * 2;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        copyGatesToGlobal(hostGates, gates.size(), MyGlobalVars::streams[g], g);
        launchDMExecutor(gridDim, deviceStateVec[g], threadBias[g], numLocalQubits, gates.size(), blockHot, enumerate, 
        MyGlobalVars::streams[g], g);
        // launchDMExecutorSerial(deviceStateVec[g], numLocalQubits, gates);
    }
}

void CudaDMExecutor::prepareBitMap(idx_t relatedQubits, unsigned int& blockHot, unsigned int& enumerate, int numLocalQubits) {
    // for data fetch & save. almost the same with state vector simulation
    unsigned int related2 = duplicate_bit(relatedQubits);
    blockHot = (idx_t(1) << (numLocalQubits * 2)) - 1 - related2;
    enumerate = related2;
    idx_t threadHot = 0;
    for (int i = 0; i < THREAD_DEP; i++) {
        idx_t x = enumerate & (-enumerate);
        threadHot += x;
        enumerate -= x;
    }
    unsigned int hostThreadBias[1 << THREAD_DEP];
    assert((threadHot | enumerate) == related2);
    for (idx_t i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
        hostThreadBias[i] = j;
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaMemcpyAsync(threadBias[g], hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
    }    
}

void CudaDMExecutor::dm_transpose() { UNIMPLEMENTED(); }
void CudaDMExecutor::transpose(std::vector<cuttHandle> plans) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaSetDevice(g);
        checkCuttErrors(cuttExecute(plans[g], deviceStateVec[g], deviceBuffer[g]));
    }
}

void CudaDMExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) { UNIMPLEMENTED(); }

void CudaDMExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit / 2;
    idx_t numElements = 1ll << (numLocalQubit * 2);
    int numPart = numSlice / commSize;
    idx_t partSize = numElements / numSlice;
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventCreate(&MyGlobalVars::events[g]));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams[g]));
    }
    for (int xr = 0; xr < commSize; xr++) {
        for (int p = 0; p < numPart; p++) {
#if USE_MPI
            checkNCCLErrors(ncclGroupStart());
#endif
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                int srcPart = a % commSize * numPart + p;
                int dstPart = b % commSize * numPart + p;
#if USE_MPI
                if (p == 0) {
                    checkCudaErrors(cudaStreamWaitEvent(
                        MyGlobalVars::streams_comm[comm_a],
                        MyGlobalVars::events[comm_a], 0)
                    );
                }
                checkCudaErrors(cudaSetDevice(comm_a));
                if (a == b) {
                    checkCudaErrors(cudaMemcpyAsync(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        deviceBuffer[comm_a] + srcPart * partSize,
                        partSize * sizeof(cpx),
                        cudaMemcpyDeviceToDevice,
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else if (a < b) {
                    checkNCCLErrors(ncclSend(
                        deviceBuffer[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclRecv(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else {
                    checkNCCLErrors(ncclRecv(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclSend(
                        deviceBuffer[comm_a] + dstPart * partSize,
                        partSize * 2, // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                }
#else
                if (p == 0) {
                    checkCudaErrors(cudaStreamWaitEvent(
                        MyGlobalVars::streams_comm[comm[a]],
                        MyGlobalVars::events[comm[b]], 0)
                    );
                }
                checkCudaErrors(cudaMemcpyAsync(
                    deviceStateVec[comm[a]] + dstPart * partSize,
                    deviceBuffer[comm[b]] + srcPart * partSize,
                    partSize * sizeof(cpx),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm[a]]
                ));
#endif
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupEnd());
#endif
        }
    }
}
void CudaDMExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) { UNIMPLEMENTED(); }
void CudaDMExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CudaDMExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CudaDMExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CudaDMExecutor::deviceFinalize() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaFree(threadBias[g]));
    }
}
void CudaDMExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CudaDMExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CudaDMExecutor::eventBarrierAll() { UNIMPLEMENTED(); }
void CudaDMExecutor::allBarrier() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams_comm[g]));
    }
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

}
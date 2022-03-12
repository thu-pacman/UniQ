#include "cuda/cuda_executor.h"

#include <chrono>
#include <assert.h>
#include <cuda_runtime.h>

#include "logger.h"
#include "cuda/kernel.h"

namespace CudaImpl {
CudaExecutor::CudaExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): Executor(deviceStateVec, numQubits, schedule) {
    threadBias.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaMalloc(&threadBias[g], sizeof(idx_t) << THREAD_DEP));
    }
}

void CudaExecutor::transpose(std::vector<cuttHandle> plans) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        cudaSetDevice(g);
        checkCuttErrors(cuttExecute(plans[g], deviceStateVec[g], deviceBuffer[g]));
    }
}

void CudaExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    idx_t oldGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        oldGlobals |= 1ll << state.layout[i];
    idx_t newGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        newGlobals |= 1ll << newState.layout[i];
    
    idx_t globalMask = 0;
    idx_t localMasks[commSize];
    idx_t localMask = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        if (newState.layout[i] != state.layout[i]) {
            int x = state.layout[i];
            globalMask |= 1ll << i;
            localMask |= 1ll << newState.pos[x];
        }

    for (idx_t i = commSize-1, msk = localMask; i >= 0; i--, msk = localMask & (msk - 1)) {
        localMasks[i] = msk;
    }

    int sliceSize = 0;
    while (sliceSize < MAX_SLICE && !(localMask >> sliceSize & 1))
        sliceSize ++;

    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams[g]));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[g], MyGlobalVars::events[g], 0));
    }

    cpx* tmpBuffer[MyGlobalVars::localGPUs];
    size_t tmpStart = 1ll << numLocalQubits;
    if (GPU_BACKEND == 3 || GPU_BACKEND == 4)
        tmpStart <<= 1;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++)
        tmpBuffer[i] = deviceStateVec[i] + tmpStart;

    for (idx_t iter = 0; iter < (1ll << numLocalQubits); iter += (1 << sliceSize)) {
        if (iter & localMask) continue;
        for (int xr = 1; xr < commSize; xr++) {
            // copy from src to tmp_buffer
            for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupStart());
#endif
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                // the (a%commSize)-th GPU in the a/commSize comm_world (comm[a]) ->
                // the (a%commSize)^xr-th GPU in the same comm_world comm[a^xr]
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t srcBias = iter + localMasks[b % commSize];
#if USE_MPI
                int comm_a = comm[a] %  MyGlobalVars::localGPUs;
                if (a < b) {
                    checkNCCLErrors(ncclSend(
                        deviceStateVec[comm_a] + srcBias,
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclRecv(
                        tmpBuffer[comm_a],
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                } else {
                    checkNCCLErrors(ncclRecv(
                        tmpBuffer[comm_a],
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                    checkNCCLErrors(ncclSend(
                        deviceStateVec[comm_a] + srcBias,
                        1 << (sliceSize + 1), // use double rather than complex
                        NCCL_FLOAT_TYPE,
                        comm[b],
                        MyGlobalVars::ncclComms[comm_a],
                        MyGlobalVars::streams_comm[comm_a]
                    ));
                }
#else
                checkCudaErrors(cudaSetDevice(comm[b]));
                checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[comm[b]], MyGlobalVars::events[comm[a]], 0));
                checkCudaErrors(cudaMemcpyAsync(
                    tmpBuffer[comm[b]],
                    deviceStateVec[comm[a]] + srcBias,
                    (sizeof(cpx) << sliceSize),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm[b]]
                ));
#endif
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupEnd());
#else
            for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
                checkCudaErrors(cudaSetDevice(g));
                checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
            }
#endif
            // copy from tmp_buffer to dst
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[b] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t dstBias = iter + localMasks[a % commSize];
                int comm_b = comm[b] % MyGlobalVars::localGPUs;
                checkCudaErrors(cudaSetDevice(comm_b));
#if not USE_MPI
                // no need to sync nccl calls, as nccl calls are synchronized.
                checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams_comm[comm_b], MyGlobalVars::events[comm[a]], 0));
#endif
                checkCudaErrors(cudaMemcpyAsync(
                    deviceStateVec[comm_b] + dstBias,
                    tmpBuffer[comm_b],
                    (sizeof(cpx) << sliceSize),
                    cudaMemcpyDeviceToDevice,
                    MyGlobalVars::streams_comm[comm_b]
                ));
            }
        }
    }
    this->eventBarrier();
}

void CudaExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit;
    idx_t numElements = 1ll << numLocalQubit;
    int numPart = numSlice / commSize;
    idx_t partSize = numElements / numSlice;
    commEvents.resize(numSlice * MyGlobalVars::localGPUs);
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    int sliceID = 0;
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
                partID[sliceID * MyGlobalVars::localGPUs + comm_a] = dstPart;
                peer[sliceID * MyGlobalVars::localGPUs + comm_a] = comm[b];
            }
#if USE_MPI
            checkNCCLErrors(ncclGroupEnd());
#endif
            // events should be recorded after ncclGroupEnd
#ifdef ENABLE_OVERLAP
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                if (USE_MPI && comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                cudaEvent_t event;
                checkCudaErrors(cudaSetDevice(comm_a));
                checkCudaErrors(cudaEventCreate(&event));
                checkCudaErrors(cudaEventRecord(event, MyGlobalVars::streams_comm[comm_a]));
                commEvents[sliceID * MyGlobalVars::localGPUs + comm_a] = event;
            }
#endif
            sliceID++;
        }
    }
#ifndef ENABLE_OVERLAP
    this->eventBarrierAll();
#endif
}

void CudaExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    unsigned int blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate, numLocalQubits);
    idx_t gridDim = (idx_t(1) << numLocalQubits) >> LOCAL_QUBIT_SIZE;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        copyGatesToSymbol(hostGates, gates.size(), MyGlobalVars::streams[g], g);
        launchExecutor(gridDim, deviceStateVec[g], threadBias[g], numLocalQubits, gates.size(), blockHot, enumerate, MyGlobalVars::streams[g], g);
    }
}

void CudaExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) {
    idx_t partSize = idx_t(1) << numLocalQubits;
    unsigned int blockHot, enumerate;
    prepareBitMap(relatedQubits, blockHot, enumerate, numLocalQubits);
    idx_t gridDim = (idx_t(1) << numLocalQubits) >> LOCAL_QUBIT_SIZE;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        copyGatesToSymbol(hostGates, gates.size(), MyGlobalVars::streams[g], g);
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        launchExecutor(gridDim, deviceStateVec[g] + pID * partSize, threadBias[g], numLocalQubits, gates.size(), blockHot, enumerate, MyGlobalVars::streams[g], g);
    }
}

void CudaExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) {
    idx_t numElements = idx_t(1) << numLocalQubits;
    cpx alpha = cpx(1.0, 0.0), beta = cpx(0.0, 0.0);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCuttErrors(cuttExecute(gg.transPlans[g], deviceStateVec[g], deviceBuffer[g]));
        int K = 1 << gg.matQubit;
        checkBlasErrors(cublasGEMM(MyGlobalVars::blasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N,
            K, numElements / K, K, // M, N, K
            reinterpret_cast<cuCpx*>(&alpha), reinterpret_cast<cuCpx*>(gg.deviceMats[g]), K, // alpha, a, lda
            reinterpret_cast<cuCpx*>(deviceBuffer[g]), K, // b, ldb
            reinterpret_cast<cuCpx*>(&beta), reinterpret_cast<cuCpx*>(deviceStateVec[g]), K // beta, c, ldc
        ));
    }
}

void CudaExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) {
    idx_t numElements = idx_t(1) << numLocalQubits;
    cpx alpha = cpx(1.0, 0.0), beta = cpx(0.0, 0.0);
    idx_t partSize = idx_t(1) << numLocalQubits;
    int K = 1 << gg.matQubit;
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        checkCuttErrors(cuttExecute(gg.transPlans[g], deviceStateVec[g] + partSize * pID, deviceBuffer[g] + partSize * pID));
        checkBlasErrors(cublasGEMM(MyGlobalVars::blasHandles[g], CUBLAS_OP_N, CUBLAS_OP_N,
            K, numElements / K, K, // M, N, K
            reinterpret_cast<cuCpx*>(&alpha), reinterpret_cast<cuCpx*>(gg.deviceMats[g]), K, // alpha, a, lda
            reinterpret_cast<cuCpx*>(deviceBuffer[g] + partSize * pID), K, // b, ldb
            reinterpret_cast<cuCpx*>(&beta), reinterpret_cast<cuCpx*>(deviceStateVec[g] + partSize * pID), K // beta, c, ldc
        ));
    }
}

void CudaExecutor::prepareBitMap(idx_t relatedQubits, unsigned int& blockHot, unsigned int& enumerate, int numLocalQubits) {
    blockHot = (idx_t(1) << numLocalQubits) - 1 - relatedQubits;
    enumerate = relatedQubits;
    idx_t threadHot = 0;
    for (int i = 0; i < THREAD_DEP; i++) {
        idx_t x = enumerate & (-enumerate);
        threadHot += x;
        enumerate -= x;
    }
    unsigned int hostThreadBias[1 << THREAD_DEP];
    assert((threadHot | enumerate) == relatedQubits);
    for (idx_t i = (1 << THREAD_DEP) - 1, j = threadHot; i >= 0; i--, j = threadHot & (j - 1)) {
        hostThreadBias[i] = j;
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaMemcpyAsync(threadBias[g], hostThreadBias, sizeof(hostThreadBias), cudaMemcpyHostToDevice, MyGlobalVars::streams[g]));
    }
}

void CudaExecutor::deviceFinalize() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaFree(threadBias[g]));
    }
}

void CudaExecutor::sliceBarrier(int sliceID) {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], commEvents[sliceID * MyGlobalVars::localGPUs + g], 0));
#if !USE_MPI
        int peerID = peer[sliceID * MyGlobalVars::localGPUs + g];
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], commEvents[sliceID * MyGlobalVars::localGPUs + peerID], 0));
#endif
    }
}

void CudaExecutor::allBarrier() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams[g]));
        checkCudaErrors(cudaStreamSynchronize(MyGlobalVars::streams_comm[g]));
    }
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void CudaExecutor::eventBarrierAll() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
    }
    for (int gg = 0; gg < MyGlobalVars::localGPUs; gg++) {
        checkCudaErrors(cudaSetDevice(gg));
        for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
            checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[gg], MyGlobalVars::events[g], 0));
        }
    }
}

void CudaExecutor::eventBarrier() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        checkCudaErrors(cudaEventRecord(MyGlobalVars::events[g], MyGlobalVars::streams_comm[g]));
        checkCudaErrors(cudaStreamWaitEvent(MyGlobalVars::streams[g], MyGlobalVars::events[g], 0));
    }
}

void CudaExecutor::dm_transpose() {
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        packing(numQubits, deviceStateVec[g], deviceBuffer[g]);
    }
    allBarrier();
    auto start = std::chrono::system_clock::now();
#if USE_MPI
    idx_t partSize = idx_t(1) << (numQubits - 2 * MyGlobalVars::bit);
    checkNCCLErrors(ncclGroupStart());
    for (int xr = 0; xr < MyGlobalVars::numGPUs; xr++) {
        for (int a = 0; a < MyGlobalVars::localGPUs; a++) {
            int comm_a = a + MyMPI::rank * MyGlobalVars::localGPUs;
            int comm_b = comm_a ^ xr;
            checkCudaErrors(cudaSetDevice(a));
            if (comm_a == comm_b) {
                checkCudaErrors(cudaMemcpyAsync(deviceStateVec[a] + comm_a * partSize, deviceBuffer[a] + comm_a * partSize, partSize * sizeof(cpx), cudaMemcpyDeviceToDevice));
            } else if (comm_a < comm_b) {
                checkNCCLErrors(ncclSend(
                    deviceBuffer[a] + comm_b * partSize,
                    partSize * 2, // use double rather than complex
                    NCCL_FLOAT_TYPE,
                    comm_b,
                    MyGlobalVars::ncclComms[a],
                    MyGlobalVars::streams_comm[a]
                ));
                checkNCCLErrors(ncclRecv(
                    deviceStateVec[a] + comm_b * partSize,
                    partSize * 2, // use double rather than complex
                    NCCL_FLOAT_TYPE,
                    comm_b,
                    MyGlobalVars::ncclComms[a],
                    MyGlobalVars::streams_comm[a]
                ));
            } else {
                checkNCCLErrors(ncclRecv(
                    deviceStateVec[a] + comm_b * partSize,
                    partSize * 2, // use double rather than complex
                    NCCL_FLOAT_TYPE,
                    comm_b,
                    MyGlobalVars::ncclComms[a],
                    MyGlobalVars::streams_comm[a]
                ));
                checkNCCLErrors(ncclSend(
                    deviceBuffer[a] + comm_b * partSize,
                    partSize * 2, // use double rather than complex
                    NCCL_FLOAT_TYPE,
                    comm_b,
                    MyGlobalVars::ncclComms[a],
                    MyGlobalVars::streams_comm[a]
                ));
            }
        }
    }
    checkNCCLErrors(ncclGroupEnd());
    allBarrier();
#else
    UNIMPLEMENTED();
#endif
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    Logger::add("All2all Time: %d us", int(duration.count()));
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        checkCudaErrors(cudaSetDevice(g));
        unpacking(numQubits, deviceStateVec[g], deviceBuffer[g]);
        // checkCudaErrors(cudaMemcpyAsync(deviceStateVec[g], deviceBuffer[g], sizeof(cpx) << numLocalQubits, cudaMemcpyDeviceToDevice));
    }
}

}
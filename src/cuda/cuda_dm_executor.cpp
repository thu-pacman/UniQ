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
void CudaDMExecutor::transpose(std::vector<cuttHandle> plans) { UNIMPLEMENTED(); }
void CudaDMExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) { UNIMPLEMENTED(); }
void CudaDMExecutor::all2all(int commSize, std::vector<int> comm) { UNIMPLEMENTED(); }
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
void CudaDMExecutor::allBarrier() { UNIMPLEMENTED(); }

}
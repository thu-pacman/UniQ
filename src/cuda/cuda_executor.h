#pragma once
#include "executor.h"

#include <cuda_runtime.h>

namespace CudaImpl {
class CudaExecutor: public Executor {
public:
    CudaExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void dm_transpose();

protected:
    void transpose(std::vector<cuttHandle> plans);
    void inplaceAll2All(int commSize, std::vector<int> comm, const State& newState);
    void all2all(int commSize, std::vector<int> comm);
    void launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits);
    void launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID);
    void launchBlasGroup(GateGroup& gg, int numLocalQubits);
    void launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID);
    void prepareBitMap(idx_t relatedQubits, unsigned int& blockHot, unsigned int& threadBias, int numLocalQubits); // allocate threadBias
    void deviceFinalize();
    void sliceBarrier(int sliceID);
    void eventBarrier();
    void eventBarrierAll();
    void allBarrier();

    std::vector<cudaEvent_t> commEvents; // commEvents[slice][gpuID]
    std::vector<unsigned int*> threadBias;
};
}
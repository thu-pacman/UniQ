#pragma once
#include "executor.h"

#include <cuda_runtime.h>

namespace CudaImpl {
class CudaExecutor: public Executor {
public:
    CudaExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void dm_transpose() override;

protected:
    void transpose(std::vector<cuttHandle> plans) override;
    void inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) override;
    void all2all(int commSize, std::vector<int> comm) override;
    void launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) override;
    void launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) override;
    void launchBlasGroup(GateGroup& gg, int numLocalQubits) override;
    void launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) override;
    void deviceFinalize() override;
    void sliceBarrier(int sliceID) override;
    void eventBarrier() override;
    void eventBarrierAll() override;
    void allBarrier() override;

    void prepareBitMap(idx_t relatedQubits, unsigned int& blockHot, unsigned int& threadBias, int numLocalQubits); // allocate threadBias
    std::vector<cudaEvent_t> commEvents; // commEvents[slice][gpuID]
    std::vector<unsigned int*> threadBias;
};
}
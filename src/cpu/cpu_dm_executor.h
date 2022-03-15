#pragma once
#include "dm_executor.h"

namespace CpuImpl {
class CpuDMExecutor: public DMExecutor {
public:
    CpuDMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void dm_transpose() override;

protected:
    void launchPerGateGroupDM(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) override;

    // unimplemented
    void transpose(std::vector<std::shared_ptr<hptt::Transpose<cpx>>> plans) override;
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
};

}

#pragma once
#include "executor.h"
#include "hptt.h"

namespace CpuImpl {
class CpuExecutor: public Executor {
public:
    CpuExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void dm_transpose();

protected:
    void transpose(std::vector<hptt::Transpose<cpx>> plans);
    void inplaceAll2All(int commSize, std::vector<int> comm, const State& newState);
    void all2all(int commSize, std::vector<int> comm);
    void launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits);
    void launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID);
    void launchBlasGroup(GateGroup& gg, int numLocalQubits);
    void launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID);
    void deviceFinalize();
    void sliceBarrier(int sliceID);
    void eventBarrier();
    void eventBarrierAll();
    void allBarrier();
};
}
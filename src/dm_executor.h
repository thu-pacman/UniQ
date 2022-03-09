#pragma once
#include "executor.h"

class DMExecutor: public Executor {
public:
    DMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void run();
protected:
    void applyPerGateGroup(GateGroup& gg) override;
    virtual void launchPerGateGroupDM(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) = 0;
};
#include "dm_executor.h"
#include <assert.h>

DMExecutor::DMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule):
    Executor(deviceStateVec, numQubits, schedule) {}

void DMExecutor::run() {
    // NOT MODIFIED
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID ++) {
        auto& localGroup = schedule.localGroups[lgID];
        if (lgID > 0) {
            if (INPLACE) {
                this->inplaceAll2All(localGroup.a2aCommSize, localGroup.a2aComm, localGroup.state);
            } else {
                // auto tag1 = std::chrono::system_clock::now();
                this->transpose(localGroup.transPlans);
                // auto tag2 = std::chrono::system_clock::now();
                this->all2all(localGroup.a2aCommSize, localGroup.a2aComm);
                // auto tag3 = std::chrono::system_clock::now();
                // Logger::add("comm: transpose %d us all2all %d us\n", (int) std::chrono::duration_cast<std::chrono::microseconds>(tag2 - tag1).count(), (int) std::chrono::duration_cast<std::chrono::microseconds>(tag3 - tag2).count());
            }
            this->allBarrier();
            this->setState(localGroup.state);
            if (schedule.localGroups[lgID].overlapGroups.size() > 0) {
                UNIMPLEMENTED();
            }
        } else {
            this->setState(localGroup.state);
            assert(localGroup.overlapGroups.size() == 0);
        }
        for (auto& gg: schedule.localGroups[lgID].fullGroups) {
            this->applyGateGroup(gg, -1);
        }
    }
    this->finalize();
}

void DMExecutor::applyPerGateGroup(GateGroup& gg) {
    auto& gates = gg.gates;
    int numLocalQubits = numQubits - MyGlobalVars::bit / 2;
    idx_t relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    idx_t relatedQubits = toPhyQubitSet(relatedLogicQb);
    std::map<int, int> toID = getLogicShareMap(relatedQubits, numLocalQubits);

    KernelGate hostGates[MyGlobalVars::localGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(MyGlobalVars::localGPUs)
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + g;
        for (size_t i = 0; i < gates.size(); i++) {
            hostGates[g * gates.size() + i] = getGate(gates[i], globalGPUID, numLocalQubits, relatedLogicQb, toID);
            addError(&hostGates[g * gates.size() + i], gates[i]);
        }
    }
    launchPerGateGroupDM(gates, hostGates, state, relatedQubits, numLocalQubits);
}


void DMExecutor::addError(KernelGate* gate, const Gate& g) {
#if MODE == 2
    gate->err_len_control = g.controlErrors.size();
    for (int i = 0; i < gate->err_len_control; i++) {
        gate->errs_control[i][0][0] = g.controlErrors[i].mat00;
        gate->errs_control[i][0][1] = g.controlErrors[i].mat01;
        gate->errs_control[i][1][0] = g.controlErrors[i].mat10;
        gate->errs_control[i][1][1] = g.controlErrors[i].mat11;
    }
    gate->err_len_target = g.targetErrors.size();
    for (int i = 0; i < gate->err_len_target; i++) {
        gate->errs_target[i][0][0] = g.targetErrors[i].mat00;
        gate->errs_target[i][0][1] = g.targetErrors[i].mat01;
        gate->errs_target[i][1][0] = g.targetErrors[i].mat10;
        gate->errs_target[i][1][1] = g.targetErrors[i].mat11;
    }
#else
    UNREACHABLE();
#endif
}
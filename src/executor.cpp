#include "executor.h"

#include <algorithm>
#include <chrono>

#include "utils.h"
#include "assert.h"
#ifdef USE_GPU
#include "cuda/entry.h"
#endif
#include "dbg.h"
#include "logger.h"

Executor::Executor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule):
    deviceStateVec(deviceStateVec),
    numQubits(numQubits),
    schedule(schedule) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    idx_t numElements = idx_t(1) << numLocalQubits;
    deviceBuffer.resize(MyGlobalVars::localGPUs);
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        deviceBuffer[g] = deviceStateVec[g] + numElements;        
    }
    numSlice = MyGlobalVars::numGPUs;
    numSliceBit = MyGlobalVars::bit;
    // TODO
    // initialize pos
}

void Executor::run() {
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID ++) {
        auto& localGroup = schedule.localGroups[lgID];
        if (lgID > 0) {
            if (INPLACE) {
                this->inplaceAll2All(localGroup.a2aCommSize, localGroup.a2aComm, localGroup.state);
            } else {
                this->transpose(localGroup.transPlans);
                this->all2all(localGroup.a2aCommSize, localGroup.a2aComm);
            }
            this->setState(localGroup.state);
#ifdef ENABLE_OVERLAP
            this->storeState();
            for (int s = 0; s < numSlice; s++) {
                this->loadState();
                this->sliceBarrier(s);
                for (auto& gg: schedule.localGroups[lgID].overlapGroups) {
                    this->applyGateGroup(gg, s);
                }
            }
#endif
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

#define SET_GATE_TO_ID(g, i) { \
    cpx mat[2][2] = {1, 0, 0, 1}; \
    hostGates[g * gates.size() + i] = KernelGate(GateType::ID, 0, 0, mat); \
}

#define IS_SHARE_QUBIT(logicIdx) ((relatedLogicQb >> logicIdx & 1) > 0)
#define IS_LOCAL_QUBIT(logicIdx) (state.pos[logicIdx] < numLocalQubits)
#define IS_HIGH_PART(part_id, logicIdx) ((part_id >> (state.pos[logicIdx] - numLocalQubits) & 1) > 0)

KernelGate Executor::getGate(const Gate& gate, int part_id, int numLocalQubits, idx_t relatedLogicQb, const std::map<int, int>& toID) const {
    if (gate.controlQubit2 != -1) {
        // Assume no CC-Diagonal
        int c1 = gate.controlQubit;
        int c2 = gate.controlQubit2;
        if (IS_LOCAL_QUBIT(c2) && !IS_LOCAL_QUBIT(c1)) {
            int c = c1; c1 = c2; c2 = c;
        }
        if (IS_LOCAL_QUBIT(c1) && IS_LOCAL_QUBIT(c2)) { // CCU(c1, c2, t)
            if (IS_SHARE_QUBIT(c2) && !IS_SHARE_QUBIT(c1)) {
                int c = c1; c1 = c2; c2 = c;
            }
            return KernelGate(
                gate.type,
                toID.at(c2), 1 - IS_SHARE_QUBIT(c2),
                toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                gate.mat
            );
        } else if (IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)) {
            if (IS_HIGH_PART(part_id, c2)) { // CU(c1, t)
                return KernelGate(
                    Gate::toCU(gate.type),
                    toID.at(c1), 1 - IS_SHARE_QUBIT(c1),
                    toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                    gate.mat
                );
            } else { // ID(t)
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c1) && !IS_LOCAL_QUBIT(c2)
            if (IS_HIGH_PART(part_id, c1) && IS_HIGH_PART(part_id, c2)) { // U(t)
                return KernelGate(
                    Gate::toU(gate.type),
                    toID.at(gate.targetQubit), 1 - IS_SHARE_QUBIT(gate.targetQubit),
                    gate.mat
                );
            } else { // ID(t)
                return KernelGate::ID();
            }
        }
    } else if (gate.controlQubit != -1) {
        int c = gate.controlQubit, t = gate.targetQubit;
        if (IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) { // CU(c, t)
            return KernelGate(
                gate.type,
                toID.at(c), 1 - IS_SHARE_QUBIT(c),
                toID.at(t), 1 - IS_SHARE_QUBIT(t),
                gate.mat
            );
        } else if (IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)) { // U(c)
            switch (gate.type) {
                case GateType::CZ: {
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            GateType::Z,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            gate.mat
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::CU1: {
                    if (IS_HIGH_PART(part_id, t)) {
                        return KernelGate(
                            GateType::U1,
                            toID.at(c), 1 - IS_SHARE_QUBIT(c),
                            gate.mat
                        );
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::CRZ: { // GOC(c)
                    cpx mat[2][2] = {cpx(1), cpx(0), cpx(0), IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0]};
                    return KernelGate(
                        GateType::GOC,
                        toID.at(c), 1 - IS_SHARE_QUBIT(c),
                        mat
                    );
                }
                default: {
                    UNREACHABLE()
                }
            }
        } else if (!IS_LOCAL_QUBIT(c) && IS_LOCAL_QUBIT(t)) {
            if (IS_HIGH_PART(part_id, c)) { // U(t)
                return KernelGate(
                    Gate::toU(gate.type),
                    toID.at(t), 1 - IS_SHARE_QUBIT(t),
                    gate.mat
                );
            } else {
                return KernelGate::ID();
            }
        } else { // !IS_LOCAL_QUBIT(c) && !IS_LOCAL_QUBIT(t)
            assert(gate.isDiagonal());
            if (IS_HIGH_PART(part_id, c)) {
                switch (gate.type) {
                    case GateType::CZ: {
                        if (IS_HIGH_PART(part_id, t)) {
                            cpx mat[2][2] = {cpx(-1), cpx(0), cpx(0), cpx(-1)};
                            return KernelGate(
                                GateType::GZZ,
                                0, 0,
                                mat
                            );
                        } else {
                            return KernelGate::ID();
                        }
                    }
                    case GateType::CU1: {
                        if (IS_HIGH_PART(part_id, t)) {
                            cpx mat[2][2] = {gate.mat[1][1], cpx(0), cpx(0), gate.mat[1][1]};
                            return KernelGate(
                                GateType::GCC,
                                0, 0,
                                mat
                            );
                        }
                    }
                    case GateType::CRZ: {
                        cpx val = IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0];
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(
                            GateType::GCC,
                            0, 0,
                            mat
                        );
                    }
                    default: {
                        UNREACHABLE()
                    }
                }
            } else {
                return KernelGate::ID();
            }
        }
    } else {
        int t = gate.targetQubit;
        if (!IS_LOCAL_QUBIT(t)) { // GCC(t)
            switch (gate.type) {
                case GateType::U1: {
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx val = gate.mat[1][1];
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::Z: {
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx mat[2][2] = {cpx(-1), cpx(0), cpx(0), cpx(-1)};
                        return KernelGate(GateType::GZZ, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::S: {
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx val = cpx(0, 1);
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(GateType::GII, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::SDG: {
                    // FIXME
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx val = cpx(0, -1);
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::T: {
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx val = gate.mat[1][1];
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::TDG: {
                    if (IS_HIGH_PART(part_id, t)) {
                        cpx val = gate.mat[1][1];
                        cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                        return KernelGate(GateType::GCC, 0, 0, mat);
                    } else {
                        return KernelGate::ID();
                    }
                }
                case GateType::RZ: {
                    cpx val = IS_HIGH_PART(part_id, t) ? gate.mat[1][1]: gate.mat[0][0];
                    cpx mat[2][2] = {val, cpx(0), cpx(0), val};
                    return KernelGate(GateType::GCC, 0, 0, mat);
                }
                case GateType::ID: {
                    return KernelGate::ID();
                }
                default: {
                    UNREACHABLE()
                }
            }
        } else { // IS_LOCAL_QUBIT(t) -> U(t)
            return KernelGate(gate.type, toID.at(t), 1 - IS_SHARE_QUBIT(t), gate.mat);
        }
    }
}

void Executor::applyGateGroup(GateGroup& gg, int sliceID) {
    switch (gg.backend) {
        case Backend::PerGate: {
            if (sliceID == -1) {
                applyPerGateGroup(gg);
            } else {
                applyPerGateGroupSliced(gg, sliceID);
            }
            break;
        }
        case Backend::BLAS: {
            if (sliceID == -1) {
                applyBlasGroup(gg);
            } else {
                applyBlasGroupSliced(gg, sliceID);
            }
            break;
        }
        default:
            UNREACHABLE()
    }
    setState(gg.state);
    // printf("Group End\n");
}

void Executor::applyPerGateGroup(GateGroup& gg) {
    auto& gates = gg.gates;
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    // initialize blockHot, enumerate, threadBias
    idx_t relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    idx_t relatedQubits = toPhyQubitSet(relatedLogicQb);
    
    // initialize gates
    std::map<int, int> toID = getLogicShareMap(relatedQubits, numLocalQubits);
    
    KernelGate hostGates[MyGlobalVars::localGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);
    #pragma omp parallel for num_threads(MyGlobalVars::localGPUs)
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + g;
        for (size_t i = 0; i < gates.size(); i++) {
           hostGates[g * gates.size() + i] = getGate(gates[i], globalGPUID, numLocalQubits, relatedLogicQb, toID);
        }
    }
    launchPerGateGroup(gates, hostGates, state, relatedQubits, numLocalQubits);
}

void Executor::applyPerGateGroupSliced(GateGroup& gg, int sliceID) {
    auto& gates = gg.gates;
    int numLocalQubits = numQubits - 2 * MyGlobalVars::bit;

    idx_t relatedLogicQb = gg.relatedQubits;
    if (bitCount(relatedLogicQb) < LOCAL_QUBIT_SIZE) {
        relatedLogicQb = fillRelatedQubits(relatedLogicQb);
    }
    idx_t relatedQubits = toPhyQubitSet(relatedLogicQb);

    // initialize gates
    std::map<int, int> toID = getLogicShareMap(relatedQubits, numLocalQubits);
    
    KernelGate hostGates[MyGlobalVars::localGPUs * gates.size()];
    assert(gates.size() < MAX_GATE);

    int numSlice = MyGlobalVars::numGPUs;
    #pragma omp parallel for num_threads(MyGlobalVars::localGPUs)
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        int pID = partID[sliceID * MyGlobalVars::localGPUs + g];
        int globalGPUID = MyMPI::rank * MyGlobalVars::localGPUs + g;
        for (size_t i = 0; i < gates.size(); i++) {
            hostGates[g * gates.size() + i] = getGate(gates[i], globalGPUID * numSlice + pID, numLocalQubits, relatedLogicQb, toID);
        }
    }

    launchPerGateGroupSliced(gates, hostGates, relatedQubits, numLocalQubits, sliceID);
}

void Executor::applyBlasGroup(GateGroup& gg) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
#ifdef OVERLAP_MAT
    gg.initMatrix(numLocalQubits);
#endif
    launchBlasGroup(gg, numLocalQubits);
}

void Executor::applyBlasGroupSliced(GateGroup& gg, int sliceID) {
    int numLocalQubits = numQubits - 2 * MyGlobalVars::bit;
    // qubits at position [n - 2 bit, n - bit) should be excluded by the compiler
#ifdef OVERLAP_MAT
    if(sliceID == 0)
        gg.initMatrix(numQubits - MyGlobalVars::bit);
#endif
    launchBlasGroupSliced(gg, numLocalQubits, sliceID);
}

idx_t Executor::toPhyQubitSet(idx_t logicQubitset) const {
     idx_t ret = 0;
    for (int i = 0; i < numQubits; i++)
        if (logicQubitset >> i & 1)
            ret |= idx_t(1) << state.pos[i];
    return ret;
}

idx_t Executor::fillRelatedQubits(idx_t relatedLogicQb) const {
    int cnt = bitCount(relatedLogicQb);
    for (int i = 0; i < LOCAL_QUBIT_SIZE; i++) {
        if (!(relatedLogicQb & (1ll << state.layout[i]))) {
            cnt++;
            relatedLogicQb |= (1ll << state.layout[i]);
            if (cnt == LOCAL_QUBIT_SIZE)
                break;
        }
    }
    return relatedLogicQb;
}

std::map<int, int> Executor::getLogicShareMap(idx_t relatedQubits, int numLocalQubits) const{
    int shareCnt = 0;
    int localCnt = 0;
    int globalCnt = 0;
    std::map<int, int> toID;
#if GPU_BACKEND==2
    for (int i = 0; i < numLocalQubits; i++)
        toID[state.layout[i]] = localCnt++;
    for (int i = numLocalQubits; i < numQubits; i++)
        toID[state.layout[i]] = globalCnt++;
#else
    for (int i = 0; i < numLocalQubits; i++) {
        if (relatedQubits & (idx_t(1) << i)) {
            toID[state.layout[i]] = shareCnt++;
        } else {
            toID[state.layout[i]] = localCnt++;
        }
    }
    for (int i = numLocalQubits; i < numQubits; i++)
        toID[state.layout[i]] = globalCnt++;
#endif
    return toID;
}

void Executor::finalize() {
    deviceFinalize();
    schedule.finalState = state;
}

void Executor::storeState() {
    oldState = state;
}

void Executor::loadState() {
    state = oldState;
}

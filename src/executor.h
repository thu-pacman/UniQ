#pragma once
#include "utils.h"

#include <vector>
#include <map>

#include "schedule.h"

class Executor {
public:
    Executor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule);
    void run();
    virtual void dm_transpose() = 0;

protected:
    // instructions
    virtual void transpose(std::vector<transHandle> plans) = 0;
    virtual void inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) = 0;
    virtual void all2all(int commSize, std::vector<int> comm) = 0;
    virtual void launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) = 0;
    virtual void launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) = 0;
    virtual void launchBlasGroup(GateGroup& gg, int numLocalQubits) = 0;
    virtual void launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) = 0;
    virtual void deviceFinalize() = 0;
    virtual void sliceBarrier(int sliceID) = 0;
    virtual void eventBarrier() = 0;
    virtual void eventBarrierAll() = 0;
    virtual void allBarrier() = 0;

    void setState(const State& newState) { state = newState; }
    void applyGateGroup(GateGroup& gg, int sliceID = -1);
    virtual void applyPerGateGroup(GateGroup& gg);
    void applyBlasGroup(GateGroup& gg);
    void applyPerGateGroupSliced(GateGroup& gg, int sliceID);
    void applyBlasGroupSliced(GateGroup& gg, int sliceID);
    void finalize();
    void storeState();
    void loadState();

    // utils
    idx_t toPhyQubitSet(idx_t logicQubitset) const;
    idx_t fillRelatedQubits(idx_t related) const;
    KernelGate getGate(const Gate& gate, int part_id, int numLocalQubits, idx_t relatedLogicQb, const std::map<int, int>& toID) const;

    // internal
    std::map<int, int> getLogicShareMap(idx_t relatedQubits, int numLocalQubits) const; // input: physical, output logic -> share

    State state;
    State oldState;
    std::vector<int> partID; // partID[slice][gpuID]
    std::vector<int> peer; // peer[slice][gpuID]

    // constants
    std::vector<cpx*> deviceStateVec;
    std::vector<cpx*> deviceBuffer;
    int numQubits;
    int numSlice, numSliceBit;

    //schedule
    Schedule& schedule;
};
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "utils.h"
#include "gate.h"

enum class Backend {
    None, PerGate, BLAS
};

std::string to_string(Backend b);

#if USE_GPU
#include <cutt.h>
typedef cuttHandle transHandle;
#elif USE_CPU
#include <hptt.h>
typedef std::shared_ptr<hptt::Transpose<cpx>> transHandle;
#else
QAQ // compile error
#endif

struct State {
    std::vector<int> pos;
    std::vector<int> layout;
    State() = default;
    State(const State&) = default;
    State(const std::vector<int>& p, const std::vector<int>& l): pos(p), layout(l) {};
    State(int numQubits) {
        pos.clear();
        for (int i = 0; i < numQubits; i++) {
            pos.push_back(i);
        }
        layout.clear();
        for (int i = 0; i < numQubits; i++) {
            layout.push_back(i);
        }
    }

    std::vector<unsigned char> serialize() const;
    static State deserialize(const unsigned char* arr, int& cur);
};

struct GateGroup {
    std::vector<Gate> gates;
    idx_t relatedQubits;
    State state;
    std::vector<int> cuttPerm;
    int matQubit;
    Backend backend;

    std::vector<transHandle> transPlans;

    std::vector<std::unique_ptr<cpx[]>> matrix;
    std::vector<cpx*> deviceMats;

    GateGroup(GateGroup&&) = default;
    GateGroup& operator = (GateGroup&&) = default;
    GateGroup(): relatedQubits(0) {}
    GateGroup copyGates();

    static GateGroup merge(const GateGroup& a, const GateGroup& b);
    static idx_t newRelated(idx_t old, const Gate& g, idx_t localQubits, bool enableGlobal);
    void addGate(const Gate& g, idx_t localQubits, bool enableGlobal);
    
    bool contains(int i) { return (relatedQubits >> i) & 1; }
    
    std::vector<unsigned char> serialize() const;
    static GateGroup deserialize(const unsigned char* arr, int& cur);

    State initState(const State& oldState, int numLocalQubits);
    State initPerGateState(const State& oldState);
    State initBlasState(const State& oldState, int numLocalQubit);
    void initCPUMatrix(int numLocalQubit);
    void initGPUMatrix();
    void initMatrix(int numLocalQubit);
    void getCuttPlanPointers(int numLocalQubits, std::vector<transHandle*> &transPlanPointers, std::vector<int*> &transPermPointers, std::vector<int> &locals);
};

struct LocalGroup {
    State state;
    int a2aCommSize;
    std::vector<int> a2aComm;
    std::vector<int> cuttPerm;

    std::vector<GateGroup> overlapGroups;
    std::vector<GateGroup> fullGroups;
    idx_t relatedQubits;

    std::vector<transHandle> transPlans;
    
    LocalGroup() = default;
    LocalGroup(LocalGroup&&) = default;

    bool contains(int i) { return (relatedQubits >> i) & 1; }
    void getCuttPlanPointers(int numLocalQubits, std::vector<transHandle*> &transPlanPointers, std::vector<int*> &transPermPointers, std::vector<int> &locals, bool isFirstGroup = false);
    State initState(const State& oldState, int numQubits, const std::vector<int>& newGlobals, idx_t overlapGlobals, idx_t overlapRelated);
    State initFirstGroupState(const State& oldState, int numQubits, const std::vector<int>& newGlobals);
    State initStateInplace(const State& oldState, int numQubits, const std::vector<int>& newGlobals, idx_t overlapGlobals);
    std::vector<unsigned char> serialize() const;
    static LocalGroup deserialize(const unsigned char* arr, int& cur);
};

struct Schedule {
    std::vector<LocalGroup> localGroups;
    State finalState;
    
    void dump(int numQubits);
    std::vector<unsigned char> serialize() const;
    static Schedule deserialize(const unsigned char* arr, int& cur);
    void initMatrix(int numQubits);
    void initCuttPlans(int numLocalQubits);
};

void removeGates(std::vector<Gate>& remain, const std::vector<Gate>& remove); // remain := remain - remove        
#pragma once

#include <string>
#include <vector>
#include "utils.h"
#include "gate.h"
#include "schedule.h"

struct ResultItem {
    ResultItem() = default;
    ResultItem(const idx_t& idx, const cpx& amp): idx(idx), amp(amp) {}
    idx_t idx;
    cpx amp;
    void print(int numQubits);
    bool operator < (const ResultItem& b) {
#if MODE == 2
        const idx_t mask = 0xaaaaaaaaaaaaaaaall;
        idx_t aa = idx & mask, bb = b.idx & mask;
        return aa == bb? (idx - aa) < (b.idx - bb) : aa < bb;
#else
        return idx < b.idx;
#endif
    }
};

class Circuit {
public:
    Circuit(int numQubits): numQubits(numQubits) {}
    void compile();
    int run(bool copy_back = true, bool destroy = true);
    void addGate(const Gate& gate) {
        gates.push_back(gate);
    }
    void dumpGates();
    void printState();
    ResultItem ampAt(idx_t idx);
    cpx ampAtGPU(idx_t idx);
    bool localAmpAt(idx_t idx, ResultItem& item);
    void add_phase_amplitude_damping_error();
    void dm_with_error();
    const int numQubits;

private:
    idx_t toPhysicalID(idx_t idx);
    idx_t toLogicID(idx_t idx);
    void masterCompile();
    void transform();
#if USE_MPI
    void gatherAndPrint(const std::vector<ResultItem>& results);
#endif
    std::vector<Gate> gates;
    std::vector<cpx*> deviceStateVec;
    std::vector<std::vector<cpx*>> deviceMats;
    Schedule schedule;
    std::vector<cpx> result;
};
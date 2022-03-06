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
    void print() {
        printf("%lld %.12f: %.12f %.12f\n", idx, amp.real() * amp.real() + amp.imag() * amp.imag(), zero_wrapper(amp.real()), zero_wrapper(amp.imag()));
    }
    bool operator < (const ResultItem& b) { return idx < b.idx; }
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
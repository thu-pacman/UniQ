#pragma once
#include <vector>
#include <set>
#include <bitset>
#include "schedule.h"
#include "utils.h"
#include "gate.h"

class Compiler {
public:
    Compiler(int numQubits, std::vector<Gate> inputGates, int globalBits);
    Schedule run();
private:
    void fillLocals(LocalGroup& lg);
    std::vector<std::pair<std::vector<Gate>, idx_t>> moveToNext(LocalGroup& lg);
    int numQubits;
    int globalBit;
    int localSize;
    int shareSize;
    bool enableGlobal;
    std::vector<Gate> gates;
};

template<int MAX_GATES>
class OneLayerCompiler {
public:
    OneLayerCompiler(int numQubits, const std::vector<Gate>& inputGates);
protected:
    int numQubits;
    std::vector<Gate> remainGates;
    std::vector<int> getGroupOpt(idx_t full, idx_t related[], bool enableGlobal, int localSize, idx_t localQubits);
    void removeGatesOpt(const std::vector<int>& remove);
    std::set<int> remain;
};

class SimpleCompiler: public OneLayerCompiler<2048> {
public:
    SimpleCompiler(int numQubits, int localSize, idx_t localQubits, const std::vector<Gate>& inputGates, bool enableGlobal, idx_t whiteList = 0, idx_t required = 0);
    LocalGroup run();
private:
    int localSize;
    idx_t localQubits;
    bool enableGlobal;
    idx_t whiteList;
    idx_t required;
};

class AdvanceCompiler: public OneLayerCompiler<512> {
public:
    AdvanceCompiler(int numQubits, idx_t localQubits, idx_t blasForbid, std::vector<Gate> inputGates, bool enableGlobal, int globalBit);
    LocalGroup run(State &state, bool usePerGate, bool useBLAS, int preGateSize, int blasSize, int cuttSize);
private:
    idx_t localQubits;
    idx_t blasForbid;
    bool enableGlobal;
    int globalBit;
};

class ChunkCompiler: public OneLayerCompiler<512> {
public:
    ChunkCompiler(int numQubits, int localSize, int chunkSize, const std::vector<Gate> &inputGates);
    LocalGroup run();
private:
    int localSize, chunkSize;
};
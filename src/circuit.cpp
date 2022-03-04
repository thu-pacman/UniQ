#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include <algorithm>
#include "utils.h"
#include "compiler.h"
#include "logger.h"
#ifdef USE_GPU
#include "cuda/cuda_executor.h"
#include "cuda/entry.h"
#endif
#ifdef USE_CPU
#include "cpu/cpu_executor.h"
#include "cpu/entry.h"
#endif
#include <cstring>
using namespace std;

#if USE_GPU
typedef CudaImpl::CudaExecutor DevExecutor;
#elif USE_CPU
typedef CpuImpl::CpuExecutor DevExecutor;
#else
TD // compile error
#endif

int Circuit::run(bool copy_back, bool destroy) {
#ifdef USE_GPU
    CudaImpl::initState(deviceStateVec, numQubits);
#elif USE_CPU
    CpuImpl::initState(deviceStateVec, numQubits);
#else
    UNIMPLEMENTED()
#endif
#ifdef USE_GPU
    CudaImpl::startProfiler();
#endif
    auto start = chrono::system_clock::now();
// #if GPU_BACKEND == 0
//     kernelExecSimple(deviceStateVec[0], numQubits, gates);
// #elif GPU_BACKEND == 1 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
#if MODE == 0
    DevExecutor(deviceStateVec, numQubits, schedule).run();
#elif MODE == 1
    DecExecutor exe1(deviceStateVec, numQubits, schedule);
    exe1.run();
    exe1.dm_transpose();
    DevExecutor exe2(deviceStateVec, numQubits, schedule);
    exe2.run();
#endif
// #elif GPU_BACKEND == 2
//     gates.clear();
//     for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID++) {
//         auto& lg = schedule.localGroups[lgID];
//         for (size_t ggID = 0; ggID < lg.overlapGroups.size(); ggID++) {
//             auto& gg = lg.overlapGroups[ggID];
//             for (auto& g: gg.gates)
//                 gates.push_back(g);
//         }
//         // if (lgID == 2) break;
//         for (size_t ggID = 0; ggID < lg.fullGroups.size(); ggID++) {
//             auto& gg = lg.fullGroups[ggID];
//             for (auto& g: gg.gates)
//                 gates.push_back(g);
//         }
//     }
//     schedule.finalState = State(numQubits);
//     kernelExecSimple(deviceStateVec[0], numQubits, gates);
// #endif
    auto end = chrono::system_clock::now();
#ifdef USE_GPU
    CudaImpl::startProfiler();
#endif
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    Logger::add("Time Cost: %d us", int(duration.count()));

    if (copy_back) {
#ifdef USE_GPU
        CudaImpl::copyBackState(result, deviceStateVec, numQubits);
#elif USE_CPU
        CpuImpl::copyBackState(result, deviceStateVec, numQubits);
#else
        UNIMPLEMENTED();
#endif
    }
    if (destroy) {
#ifdef USE_GPU
        CudaImpl::destroyState(deviceStateVec);
#elif USE_CPU
        CpuImpl::destroyState(deviceStateVec);
#else
        UNIMPLEMENTED();
#endif
    }
    return duration.count();
}

void Circuit::dumpGates() {
    int totalGates = gates.size();
    printf("total Gates: %d\n", totalGates);
    int L = 3;
    for (const Gate& gate: gates) {
        for (int i = 0; i < numQubits; i++) {
            if (i == gate.controlQubit) {
                printf(".");
                for (int j = 1; j < L; j++) printf(" ");
            } else if (i == gate.targetQubit) {
                printf("%s", gate.name.c_str());
                for (int j = gate.name.length(); j < L; j++)
                    printf(" ");
            } else {
                printf("|");
                for (int j = 1; j < L; j++) printf(" ");
            }
        }
        printf("\n");
    }
}

idx_t Circuit::toPhysicalID(idx_t idx) {
    idx_t id = 0;
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= idx_t(1) << pos[i];
    }
    return id;
}

idx_t Circuit::toLogicID(idx_t idx) {
    idx_t id = 0;
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> pos[i] & 1)
            id |= idx_t(1) << i;
    }
    return id;
}

ResultItem Circuit::ampAt(idx_t idx) {
    idx_t id = toPhysicalID(idx);
    return ResultItem(idx, result[id]);
}

cpx Circuit::ampAtGPU(idx_t idx) {
    idx_t id = toPhysicalID(idx);
    cpx ret;
#if USE_MPI
    idx_t localAmps = (1ll << numQubits) / MyMPI::commSize;
    idx_t rankID = id / localAmps;

    if (!USE_MPI || MyMPI::rank == rankID) {
        idx_t localID = id % localAmps;
#else
        idx_t localID = id;
#endif
        idx_t localGPUAmp = (1ll << numQubits) / MyGlobalVars::numGPUs;
        int gpuID = localID / localGPUAmp;
        idx_t localIdx = localID % localGPUAmp;
#ifdef USE_GPU
        ret = CudaImpl::getAmp(deviceStateVec, gpuID, localIdx);
#else
        UNIMPLEMENTED(); // not implemented
#endif
#if USE_MPI
    }
    MPI_Bcast(&ret, 1, MPI_Complex, rankID, MPI_COMM_WORLD);
#endif
    return ret;
}

bool Circuit::localAmpAt(idx_t idx, ResultItem& item) {
    idx_t localAmps = (1ll << numQubits) / MyMPI::commSize;
    idx_t id = toPhysicalID(idx);
    if (id / localAmps == MyMPI::rank) {
        // printf("%d belongs to rank %d\n", idx, MyMPI::rank);
        idx_t localID = id % localAmps;
        item = ResultItem(idx, result[localID]);
        return true;
    }
    return false;
}

void Circuit::masterCompile() {
    Logger::add("Total Gates %d", int(gates.size()));
#if GPU_BACKEND == 1 || GPU_BACKEND == 2 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
    Compiler compiler(numQubits, gates);
    schedule = compiler.run();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.fullGroups.size();
    int fullGates = 0, overlapGates = 0;
    for (auto& lg: schedule.localGroups) {
        for (auto& gg: lg.fullGroups) fullGates += gg.gates.size();
        for (auto& gg: lg.overlapGroups) overlapGates += gg.gates.size();
    }
    Logger::add("Total Groups: %d %d %d %d", int(schedule.localGroups.size()), totalGroups, fullGates, overlapGates);
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
#else
    schedule.finalState = State(numQubits);
#endif
}

void Circuit::compile() {
    auto start = chrono::system_clock::now();
    this->transform();
#if USE_MPI
    if (MyMPI::rank == 0) {
        masterCompile();
        auto s = schedule.serialize();
        int bufferSize = (int) s.size();
        checkMPIErrors(MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD));
        checkMPIErrors(MPI_Bcast(s.data(), bufferSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD));
        // int cur = 0;
        // schedule = Schedule::deserialize(s.data(), cur);
    } else {
        int bufferSize;
        checkMPIErrors(MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD));
        unsigned char* buffer = new unsigned char [bufferSize];
        checkMPIErrors(MPI_Bcast(buffer, bufferSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD));
        int cur = 0;
        schedule = Schedule::deserialize(buffer, cur);
        delete[] buffer;
        fflush(stdout);
    }
#else
    masterCompile();
#endif
    auto mid = chrono::system_clock::now();
    schedule.initCuttPlans(numQubits - MyGlobalVars::bit);
#ifndef OVERLAP_MAT
    schedule.initMatrix(numQubits);
#endif
    auto end = chrono::system_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(mid - start);
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end - mid);
    Logger::add("Compile Time: %d us + %d us = %d us", int(duration1.count()), int(duration2.count()), int(duration1.count()) + int(duration2.count()));
}

#if USE_MPI
void Circuit::gatherAndPrint(const std::vector<ResultItem>& results) {
    if (MyMPI::rank == 0) {
        int size = results.size();
        int sizes[MyMPI::commSize];
        MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int disp[MyMPI::commSize + 1];
        disp[0] = 0;
        for (int i = 0; i < MyMPI::commSize; i++)
            disp[i + 1] = disp[i] + sizes[i];
        int totalItem = disp[MyMPI::commSize];
        ResultItem* collected = new ResultItem[totalItem];
        for (int i = 0; i < MyMPI::commSize; i++)
            sizes[i] *= sizeof(ResultItem);
        for (int i = 0; i < MyMPI::commSize; i++)
            disp[i] *= sizeof(ResultItem);
        MPI_Gatherv(
            results.data(), results.size() * sizeof(ResultItem), MPI_UNSIGNED_CHAR,
            collected, sizes, disp,
            MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD
        );
        sort(collected, collected + totalItem);
        for (int i = 0; i < totalItem; i++)
            collected[i].print();
        delete[] collected;
    } else {
        int size = results.size();
        MPI_Gather(&size, 1, MPI_INT, nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(
            results.data(), results.size() * sizeof(ResultItem), MPI_UNSIGNED_CHAR,
            nullptr, nullptr, nullptr,
            MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD
        );
    }
}
#endif


void Circuit::printState() {
#if USE_MPI
    std::vector<ResultItem> results;
    ResultItem item;
    for (int i = 0; i < 128; i++) {
        if (localAmpAt(i, item)) {
            results.push_back(item);
        }
    }
    gatherAndPrint(results);
#ifdef SHOW_SCHEDULE
    results.clear();
    for (int i = 0; i < numQubits; i++) {
        if (localAmpAt(1ll << i, item)) {
            results.push_back(item);
        }
    }
    if (localAmpAt((1ll << numQubits) - 1, item)) {
        results.push_back(item);
    }
    gatherAndPrint(results);
#endif
    results.clear();
    int numLocalAmps = (1ll << numQubits) / MyMPI::commSize;
    for (idx_t i = 0; i < numLocalAmps; i++) {
        if (std::norm(result[i]) > 0.001) {
            idx_t logicID = toLogicID(i + numLocalAmps * MyMPI::rank);
            if (logicID >= 128) {
                // printf("large amp %d belongs to %d\n", logicID, MyMPI::rank);
                results.push_back(ResultItem(logicID, result[i]));
            }
        }
    }
    gatherAndPrint(results);
#else
    std::vector<ResultItem> results;
    for (int i = 0; i < 128; i++) {
        results.push_back(ampAt(i));
    }
#ifdef SHOW_SCHEDULE
    for (int i = 0; i < numQubits; i++) {
        results.push_back(ampAt(1ll << i));
    }
    results.push_back(ampAt((1ll << numQubits) - 1));
#endif
    for (auto& item: results)
        item.print();
    results.clear();
    for (idx_t i = 0; i < (1ll << numQubits); i++) {
        if (std::norm(result[i]) > 0.001) {
            idx_t logicID = toLogicID(i);
            if (logicID >= 128) {
                results.push_back(ResultItem(toLogicID(i), result[i]));
            }
        }
    }
    sort(results.begin(), results.end());
    for (auto& item: results)
        item.print();
#endif
}


// transformations

void hczh2cx(std::vector<Gate> &gates, int numQubits, bool erased[]) {
    //   .       .
    //   |       |
    // H . H =>  X
    for (int i = 0; i < (int) gates.size(); i++) {
        if (erased[i]) continue;
        Gate& gate = gates[i];
        if (gate.type == GateType::CZ) {
            int h_low_ctr = -1, h_low_tar = -1, h_high_ctr = -1, h_high_tar = -1;
            for (int j = i - 1; j >= 0; j--) {
                if (gates[j].targetQubit == gate.targetQubit) {
                    if (gates[j].type == GateType::H && !erased[j]) {
                        h_low_tar = j;
                    }
                    break;
                }
                if (gates[j].controlQubit == gate.targetQubit) break;
            }
            for (int j = i - 1; j >= 0; j--) {
                if (gates[j].targetQubit == gate.controlQubit) {
                    if (gates[j].type == GateType::H && !erased[j]) {
                        h_low_ctr = j;
                    }
                    break;
                }
                if (gates[j].controlQubit == gate.controlQubit) break;
            }
            for (int j = i + 1; j < (int) gates.size(); j++) {
                if (gates[j].targetQubit == gate.targetQubit) {
                    if (gates[j].type == GateType::H && !erased[j]) {
                        h_high_tar = j;
                    }
                    break;
                }
                if (gates[j].controlQubit == gate.targetQubit) break;
            }
            for (int j = i + 1; j < (int) gates.size(); j++) {
                if (gates[j].targetQubit == gate.controlQubit) {
                    if (gates[j].type == GateType::H && !erased[j]) {
                        h_high_ctr = j;
                    }
                    break;
                }
                if (gates[j].controlQubit == gate.controlQubit) break;
            }
            if (h_low_tar != -1 && h_high_tar != -1) {
#ifdef SHOW_SCHEDULE
                printf("[hczh2cx] %d %d %d\n", h_low_tar, i, h_high_tar);
#endif
                erased[h_low_tar] = erased[h_high_tar] = 1;
                int id = gates[i].gateID;
                gates[i] = Gate::CNOT(gate.controlQubit, gate.targetQubit);
                gates[i].gateID = id;
            } else if (h_low_ctr != -1 && h_high_ctr != -1) {
#ifdef SHOW_SCHEDULE
                printf("[hczh2cx] %d %d %d\n", h_low_ctr, i, h_high_ctr);
#endif
                erased[h_low_ctr] = erased[h_high_ctr] = 1;
                int id = gates[i].gateID;
                gates[i] = Gate::CNOT(gate.targetQubit, gate.controlQubit);
                gates[i].gateID = id;
            }
        }
    }
}

void single_qubit_fusion(std::vector<Gate> &gates, int numQubits, bool erased[]) {
    int lastGate[numQubits];
    memset(lastGate, -1, sizeof(int) * numQubits);
    for (int i = 0; i < (int) gates.size(); i++) {
        if (erased[i]) continue;
        Gate& gate = gates[i];
        int old_id = lastGate[gate.targetQubit];
        if (gate.isSingleGate() && old_id != -1 && !erased[old_id]) {
            Gate& old = gates[old_id];
            if (old.isSingleGate()) {
#ifdef SHOW_SCHEDULE
                printf("[single qubit fusion] %d %d\n", old_id, i);
#endif
                cpx mat[2][2];
                mat[0][0] = gate.mat[0][0] * old.mat[0][0] + gate.mat[0][1] * old.mat[1][0];
                mat[0][1] = gate.mat[0][0] * old.mat[0][1] + gate.mat[0][1] * old.mat[1][1];
                mat[1][0] = gate.mat[1][0] * old.mat[0][0] + gate.mat[1][1] * old.mat[1][0];
                mat[1][1] = gate.mat[1][0] * old.mat[0][1] + gate.mat[1][1] * old.mat[1][1];
                erased[old_id] = true;
                int id = gates[i].gateID;
                gates[i] = Gate::U(gate.targetQubit, {mat[0][0], mat[0][1], mat[1][0], mat[1][1]});
                gates[i].gateID = id;
            }
        }
        if (gate.isControlGate()) {
            lastGate[gate.controlQubit] = i;
        } else if (gate.isMCGate()) {
            for (auto qid: gate.controlQubits) {
                lastGate[qid] = i;
            }
        } else if (gate.isTwoQubitGate()) {
            lastGate[gate.encodeQubit] = i;
        }
        lastGate[gate.targetQubit] = i;
    }
}

inline bool eps_equal(cpx a, cpx b) {
    const value_t eps = 1e-14;
    return a.real() - b.real() < eps && b.real() - a.real() < eps && a.imag() - b.imag() < eps && b.imag() - a.imag() < eps;
}

void Circuit::transform() {
    bool *erased = new bool[gates.size()];
    memset(erased, 0, sizeof(bool) * gates.size());
    hczh2cx(this->gates, numQubits, erased);
    single_qubit_fusion(this->gates, numQubits, erased);
    std::vector<Gate> new_gates;
    for (int i = 0; i < (int) gates.size(); i++)
        if (!erased[i])
            new_gates.push_back(gates[i]);
    gates = new_gates;
    delete[] erased;
}
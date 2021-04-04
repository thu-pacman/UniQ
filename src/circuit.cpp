#include "circuit.h"

#include <cstdio>
#include <assert.h>
#include <chrono>
#include <mpi.h>
#include <algorithm>
#include <cuda_profiler_api.h>
#include "utils.h"
#include "kernel.h"
#include "compiler.h"
#include "logger.h"
#include "executor.h"
using namespace std;

int Circuit::run(bool copy_back) {
    kernelInit(deviceStateVec, numQubits);
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStart());
    }
    auto start = chrono::system_clock::now();
#if BACKEND == 0
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#elif BACKEND == 1 || BACKEND == 3 || BACKEND == 4 || BACKEND == 5
    Executor(deviceStateVec, numQubits, schedule).run();
#elif BACKEND == 2
    gates.clear();
    for (size_t lgID = 0; lgID < schedule.localGroups.size(); lgID++) {
        auto& lg = schedule.localGroups[lgID];
        for (size_t ggID = 0; ggID < lg.overlapGroups.size(); ggID++) {
            auto& gg = lg.overlapGroups[ggID];
            for (auto& g: gg.gates)
                gates.push_back(g);
        }
        if (lgID == 2) break;
        for (size_t ggID = 0; ggID < lg.fullGroups.size(); ggID++) {
            auto& gg = lg.fullGroups[ggID];
            for (auto& g: gg.gates)
                gates.push_back(g);
        }
    }
    schedule.finalState = State(numQubits);
    kernelExecSimple(deviceStateVec[0], numQubits, gates);
#endif
    auto end = chrono::system_clock::now();
    for (int i = 0; i < MyGlobalVars::localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaProfilerStop());
    }
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    Logger::add("Time Cost: %d us", int(duration.count()));
    result.resize(1ll << numQubits);
    if (copy_back) {
#if BACKEND == 0 || BACKEND == 2
        kernelDeviceToHost((qComplex*)result.data(), deviceStateVec[0], numQubits);
#else
        qindex elements = 1ll << (numQubits - MyGlobalVars::bit);
        for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
            kernelDeviceToHost((qComplex*)result.data() + elements * g, deviceStateVec[g], numQubits - MyGlobalVars::bit);
        }
#endif
    }
    for (int g = 0; g < MyGlobalVars::localGPUs; g++) {
        kernelDestroy(deviceStateVec[g]);
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

qindex Circuit::toPhysicalID(qindex idx) {
    qindex id = 0;
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= qindex(1) << pos[i];
    }
    return id;
}

qindex Circuit::toLogicID(qindex idx) {
    qindex id = 0;
    auto& pos = schedule.finalState.pos;
    for (int i = 0; i < numQubits; i++) {
        if (idx >> pos[i] & 1)
            id |= qindex(1) << i;
    }
    return id;
}

qComplex Circuit::ampAt(qindex idx) {
    qindex id = toPhysicalID(idx);
    return make_qComplex(result[id].x, result[id].y);
}

void Circuit::masterCompile() {
    auto start = chrono::system_clock::now();
    Logger::add("Total Gates %d", int(gates.size()));
#if BACKEND == 1 || BACKEND == 2 || BACKEND == 3 || BACKEND == 4 || BACKEND == 5
    Compiler compiler(numQubits, gates);
    schedule = compiler.run();
    auto mid = chrono::system_clock::now();
    int totalGroups = 0;
    for (auto& lg: schedule.localGroups) totalGroups += lg.fullGroups.size();
    int fullGates = 0, overlapGates = 0;
    for (auto& lg: schedule.localGroups) {
        for (auto& gg: lg.fullGroups) fullGates += gg.gates.size();
        for (auto& gg: lg.overlapGroups) overlapGates += gg.gates.size();
    }
    Logger::add("Total Groups: %d %d %d %d", int(schedule.localGroups.size()), totalGroups, fullGates, overlapGates);
    //schedule.initMatrix(numQubits);
#ifdef SHOW_SCHEDULE
    schedule.dump(numQubits);
#endif
#else
    auto mid = chrono::system_clock::now();
    schedule.finalState = State(numQubits);
#endif
    auto end = chrono::system_clock::now();
    auto duration1 = chrono::duration_cast<chrono::microseconds>(mid - start);
    auto duration2 = chrono::duration_cast<chrono::microseconds>(end - mid);
    Logger::add("Compile Time: %d us %d us", int(duration1.count()), int(duration2.count()));
}

void Circuit::compile() {
#if USE_MPI
    if (MyMPI::rank == 0) {
        masterCompile();
        auto s = schedule.serialize();
        int bufferSize = (int) s.size();
        checkMPIErrors(MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD));
        checkMPIErrors(MPI_Bcast(s.data(), bufferSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD));
        int cur = 0;
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
    schedule.initCuttPlans(numQubits - MyGlobalVars::bit);
}


void Circuit::printState() {
    for (int i = 0; i < 128; i++) {
        qComplex x = ampAt(i);
        printf("%d %.12f: %.12f %.12f\n", i, x.x * x.x + x.y * x.y, zero_wrapper(x.x), zero_wrapper(x.y));
    }
#ifdef SHOW_SCHEDULE
    for (int i = 0; i < numQubits; i++) {
        qComplex x = ampAt(1ll << i);
        printf("%lld %.12f: %.12f %.12f\n", 1ll << i, x.x * x.x + x.y * x.y, zero_wrapper(x.x), zero_wrapper(x.y));
    }
    qComplex y = ampAt((1ll << numQubits) - 1);
    printf("%lld %.12f: %.12f %.12f\n", (1ll << numQubits) - 1, y.x * y.x + y.y * y.y, zero_wrapper(y.x), zero_wrapper(y.y));
#endif
    std::vector<std::pair<qindex, qComplex>> largeAmps;
    for (qindex i = 0; i < (1ll << numQubits); i++) {
        if (result[i].x * result[i].x + result[i].y * result[i].y > 0.001) {
            qindex logicID = toLogicID(i);
            if (logicID >= 128) {
                largeAmps.push_back(make_pair(toLogicID(i), result[i]));
            }
        }
    }
    sort(largeAmps.begin(), largeAmps.end());
    for (auto& amp: largeAmps) {
        auto& x = amp.second;
        printf("%d %.12f: %.12f %.12f\n", amp.first, x.x * x.x + x.y * x.y, zero_wrapper(x.x), zero_wrapper(x.y));
    }
}
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
#include "cuda/cuda_dm_executor.h"
#endif
#ifdef USE_CPU
#include "cpu/cpu_executor.h"
#include "cpu/entry.h"
#include "cpu/cpu_dm_executor.h"
#endif
#include <cstring>
using namespace std;

#if USE_GPU
typedef CudaImpl::CudaExecutor DevExecutor;
#if MODE == 2
typedef CudaImpl::CudaDMExecutor DevDMExecutor;
#endif
#elif USE_CPU
typedef CpuImpl::CpuExecutor DevExecutor;
#if MODE == 2
typedef CpuImpl::CpuDMExecutor DevDMExecutor;
#endif
#else
TD // compile error
#endif

void ResultItem::print(int numQubits) {
    switch (MODE) {
        case 0: {
            printf("%lld %.12f: %.12f %.12f\n", idx, amp.real() * amp.real() + amp.imag() * amp.imag(), zero_wrapper(amp.real()), zero_wrapper(amp.imag()));
            break;
        }
        case 1: {
            printf("%lld %lld %.12f: %.12f %.12f\n", idx >> (numQubits / 2), idx & ((1 << (numQubits / 2)) - 1), amp.real() * amp.real() + amp.imag() * amp.imag(), zero_wrapper(amp.real()), zero_wrapper(amp.imag()));
            break;
        }
        case 2: {
            idx_t row = 0, col = 0;
            for (int i = 0; i < numQubits; i++) {
                int bit = idx >> i & 1;
                if (i & 1) {
                    row |= bit << (i / 2);
                } else {
                    col |= bit << (i / 2);
                }
            }
            printf("%lld %lld %.12f: %.12f %.12f\n", row, col, amp.real() * amp.real() + amp.imag() * amp.imag(), zero_wrapper(amp.real()), zero_wrapper(amp.imag()));
            break;
        }
    }
}

int Circuit::run(bool copy_back, bool destroy) {
#ifdef USE_GPU
    CudaImpl::initState(deviceStateVec, numQubits);
#elif USE_CPU
    CpuImpl::initState(deviceStateVec, numQubits);
#else
    UNIMPLEMENTED()
#endif
#if MODE == 2
    if (MyGlobalVars::bit % 2 != 0) {
        UNIMPLEMENTED();
    }
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
    DevExecutor exe1(deviceStateVec, numQubits, schedule);
    exe1.run();
#elif MODE == 2
    DevDMExecutor(deviceStateVec, numQubits / 2, schedule).run();
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
    CudaImpl::stopProfiler();
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

void Circuit::add_phase_amplitude_damping_error() {
    value_t param_amp[50] = {
        0.13522296, 0.34196305, 0.24942207, 0.20366025, 0.36708856,
        0.22573069, 0.16126925, 0.22023124, 0.19477643, 0.22062259,
        0.34242915, 0.29556578, 0.14447562, 0.24413769, 0.36841306,
        0.29977425, 0.18354474, 0.17749279, 0.35026603, 0.34237515,
        0.30820619, 0.17611002, 0.23364228, 0.20900146, 0.26849671,
        0.19429553, 0.29175968, 0.32635873, 0.20648301, 0.19582834,
        0.16577554, 0.20725059, 0.3529493 , 0.15643779, 0.13911531,
        0.13506932, 0.22451938, 0.19976538, 0.12964262, 0.34413908,
        0.35384347, 0.37235135, 0.34113041, 0.17087591, 0.28486187,
        0.35908144, 0.30639709, 0.30138282, 0.37030199, 0.12811117
    };
    value_t param_phase[50] = {
        0.27165516, 0.35525184, 0.1916562 , 0.20513042, 0.27364267,
        0.15848727, 0.37182112, 0.30637188, 0.31124254, 0.33848456,
        0.26229897, 0.12982723, 0.32468533, 0.20456679, 0.15046644,
        0.31481037, 0.33237344, 0.22990046, 0.24478173, 0.34522711,
        0.34800876, 0.27030219, 0.14428052, 0.24037756, 0.36350212,
        0.22666077, 0.27186536, 0.16700415, 0.21254885, 0.34969858,
        0.29483833, 0.25706624, 0.27592144, 0.33215269, 0.33985181,
        0.15013914, 0.27628303, 0.2027231 , 0.31656706, 0.27485518,
        0.30443711, 0.3564536 , 0.29340223, 0.19076045, 0.20382232,
        0.15499888, 0.31420134, 0.21966027, 0.24792838, 0.29566892
    };

    int n2 = numQubits / 2;
    std::vector<Error> errs[n2];
    for (int i = 0; i < n2; i++) {
        value_t amp = param_amp[i], phase = param_phase[i];
        value_t param = 1 - amp - phase;
        errs[i] = {
            Error(GateType::GOC, 1.0, 0.0, 0.0, sqrt(param)),
            Error(GateType::V01, 0.0, sqrt(amp), 0.0, 0.0),
            Error(GateType::DIG, 0.0, 0.0, 0.0, sqrt(phase))
        };
    }

    for (auto& gate: gates) {
        if (gate.isControlGate()) {
            gate.controlErrors = errs[gate.controlQubit];
            gate.targetErrors = errs[gate.targetQubit];
        } else if (gate.isSingleGate()) {
            gate.targetErrors = errs[gate.targetQubit];
        } else if (gate.isMCGate()) {
            UNIMPLEMENTED(); // you can use ID() gates for applying errors to specific qubits
        } else if (gate.isTwoQubitGate()) {
            gate.controlErrors = errs[gate.encodeQubit];
            gate.targetErrors = errs[gate.targetQubit];
        } else {
            UNREACHABLE();
        }
    }
}

struct ErrorGates {
    std::vector<Gate> gates;
};

#include <cmath>
void Circuit::dm_with_error() {
    // phase_amplitude_damping_error
    value_t param_amp[50] = {
        0.13522296, 0.34196305, 0.24942207, 0.20366025, 0.36708856,
        0.22573069, 0.16126925, 0.22023124, 0.19477643, 0.22062259,
        0.34242915, 0.29556578, 0.14447562, 0.24413769, 0.36841306,
        0.29977425, 0.18354474, 0.17749279, 0.35026603, 0.34237515,
        0.30820619, 0.17611002, 0.23364228, 0.20900146, 0.26849671,
        0.19429553, 0.29175968, 0.32635873, 0.20648301, 0.19582834,
        0.16577554, 0.20725059, 0.3529493 , 0.15643779, 0.13911531,
        0.13506932, 0.22451938, 0.19976538, 0.12964262, 0.34413908,
        0.35384347, 0.37235135, 0.34113041, 0.17087591, 0.28486187,
        0.35908144, 0.30639709, 0.30138282, 0.37030199, 0.12811117
    };
    value_t param_phase[50] = {
        0.27165516, 0.35525184, 0.1916562 , 0.20513042, 0.27364267,
        0.15848727, 0.37182112, 0.30637188, 0.31124254, 0.33848456,
        0.26229897, 0.12982723, 0.32468533, 0.20456679, 0.15046644,
        0.31481037, 0.33237344, 0.22990046, 0.24478173, 0.34522711,
        0.34800876, 0.27030219, 0.14428052, 0.24037756, 0.36350212,
        0.22666077, 0.27186536, 0.16700415, 0.21254885, 0.34969858,
        0.29483833, 0.25706624, 0.27592144, 0.33215269, 0.33985181,
        0.15013914, 0.27628303, 0.2027231 , 0.31656706, 0.27485518,
        0.30443711, 0.3564536 , 0.29340223, 0.19076045, 0.20382232,
        0.15499888, 0.31420134, 0.21966027, 0.24792838, 0.29566892
    };

    ErrorGates errs[numQubits / 2];

    // c0 = 1 c1 = 0
    for (int i = 0; i < numQubits / 2; i++) {
        value_t amp = param_amp[i], phase = param_phase[i];
        value_t param = 1 - amp - phase;
        value_t A0[2][2] = {{1.0, 0.0}, {0.0, sqrt(param)}};
        value_t A1[2][2] = {{0.0, sqrt(amp)}, {0.0, 0.0}};
        value_t A2[2][2] = {{0.0, 0.0}, {0.0, sqrt(phase)}};
        errs[i].gates = {Gate::GOC(i, A0[1][1], 0.0), Gate::V01(i, A1[0][1]), Gate::DIG(i, cpx(0.0), A2[1][1])};
    }

    cpx* state = new cpx[1 << numQubits];
    memset(reinterpret_cast<void*>(state), 0, sizeof(cpx) * (1 << numQubits));
    state[0] = cpx(1.0);
    int n2 = numQubits / 2;
    for (auto& gate: gates) {
        // U\rhoU^{\dagger}
        std::vector<int> targets;
        if (gate.isSingleGate()) {
            int t = gate.targetQubit;
            targets.push_back(t);
            idx_t mask_inner = (idx_t(1) << t) - 1;
            idx_t mask_outer = (idx_t(1) << (numQubits - 1)) - 1 - mask_inner;
            cpx mat[2][2] = {{gate.mat[0][0], gate.mat[0][1]}, {gate.mat[1][0], gate.mat[1][1]}};
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 1)); i++) {
                idx_t lo = (i & mask_inner) + ((i & mask_outer) << 1);
                idx_t hi = lo | (idx_t(1) << t);
                cpx lo_val = state[lo];
                cpx hi_val = state[hi];
                state[lo] = lo_val * mat[0][0] + hi_val * mat[0][1];
                state[hi] = lo_val * mat[1][0] + hi_val * mat[1][1];
            }
            t += n2;
            // do not need to transpose after t+=n2
            mat[0][0] = std::conj(mat[0][0]);
            mat[0][1] = std::conj(mat[0][1]);
            mat[1][0] = std::conj(mat[1][0]);
            mat[1][1] = std::conj(mat[1][1]);
            mask_inner = (idx_t(1) << t) - 1;
            mask_outer = (idx_t(1) << (numQubits - 1)) - 1 - mask_inner;
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 1)); i++) {
                idx_t lo = (i & mask_inner) + ((i & mask_outer) << 1);
                idx_t hi = lo | (idx_t(1) << t);
                cpx lo_val = state[lo];
                cpx hi_val = state[hi];
                state[lo] = lo_val * mat[0][0] + hi_val * mat[0][1];
                state[hi] = lo_val * mat[1][0] + hi_val * mat[1][1];
            }
        } else if (gate.isControlGate()) {
            int c = gate.controlQubit;
            int t = gate.targetQubit;
            targets.push_back(c);
            targets.push_back(t);
            idx_t low_bit = std::min(c, t);
            idx_t high_bit = std::max(c, t);
            idx_t mask_inner = (idx_t(1) << low_bit) - 1;
            idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
            idx_t mask_outer = (idx_t(1) << (numQubits - 2)) - 1 - mask_inner - mask_middle;
            cpx mat[2][2] = {{gate.mat[0][0], gate.mat[0][1]}, {gate.mat[1][0], gate.mat[1][1]}};
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 2)); i++) {
                idx_t lo = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                lo |= idx_t(1) << c;
                idx_t hi = lo | (idx_t(1) << t);
                cpx lo_val = state[lo];
                cpx hi_val = state[hi];
                state[lo] = lo_val * mat[0][0] + hi_val * mat[0][1];
                state[hi] = lo_val * mat[1][0] + hi_val * mat[1][1];
            }
            c += n2;
            t += n2;
            mat[0][0] = std::conj(mat[0][0]);
            mat[0][1] = std::conj(mat[0][1]);
            mat[1][0] = std::conj(mat[1][0]);
            mat[1][1] = std::conj(mat[1][1]);
            low_bit = std::min(c, t);
            high_bit = std::max(c, t);
            mask_inner = (idx_t(1) << low_bit) - 1;
            mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
            mask_outer = (idx_t(1) << (numQubits - 2)) - 1 - mask_inner - mask_middle;
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 2)); i++) {
                idx_t lo = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                lo |= idx_t(1) << c;
                idx_t hi = lo | (idx_t(1) << t);
                cpx lo_val = state[lo];
                cpx hi_val = state[hi];
                state[lo] = lo_val * mat[0][0] + hi_val * mat[0][1];
                state[hi] = lo_val * mat[1][0] + hi_val * mat[1][1];
            }
        } else if (gate.isTwoQubitGate()) {
            int c = gate.encodeQubit;
            int t = gate.targetQubit;
            targets.push_back(c);
            targets.push_back(t);
            idx_t low_bit = std::min(c, t);
            idx_t high_bit = std::max(c, t);
            idx_t mask_inner = (idx_t(1) << low_bit) - 1;
            idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
            idx_t mask_outer = (idx_t(1) << (numQubits - 2)) - 1 - mask_inner - mask_middle;
            cpx mat[2][2] = {{gate.mat[0][0], gate.mat[0][1]}, {gate.mat[1][0], gate.mat[1][1]}};
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 2)); i++) {
                idx_t s00 = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                idx_t s01 = s00 | (idx_t(1) << t);
                idx_t s10 = s00 | (idx_t(1) << c);
                idx_t s11 = s01 | s10;
                cpx v00 = state[s00];
                cpx v11 = state[s11];
                state[s00] = v00 * mat[0][0] + v11 * mat[1][1];
                state[s11] = v11 * mat[0][0] + v00 * mat[1][1];

                cpx v01 = state[s01];
                cpx v10 = state[s10];
                state[s01] = v01 * mat[0][1] + v10 * mat[1][0];
                state[s10] = v10 * mat[0][1] + v01 * mat[1][0];
            }
            c += n2;
            t += n2;
            mat[0][0] = std::conj(mat[0][0]);
            mat[0][1] = std::conj(mat[0][1]);
            mat[1][0] = std::conj(mat[1][0]);
            mat[1][1] = std::conj(mat[1][1]);
            low_bit = std::min(c, t);
            high_bit = std::max(c, t);
            mask_inner = (idx_t(1) << low_bit) - 1;
            mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
            mask_outer = (idx_t(1) << (numQubits - 2)) - 1 - mask_inner - mask_middle;
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 2)); i++) {
                idx_t s00 = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                idx_t s01 = s00 | (idx_t(1) << t);
                idx_t s10 = s00 | (idx_t(1) << c);
                idx_t s11 = s01 | s10;
                cpx v00 = state[s00];
                cpx v11 = state[s11];
                state[s00] = v00 * mat[0][0] + v11 * mat[1][1];
                state[s11] = v11 * mat[0][0] + v00 * mat[1][1];

                cpx v01 = state[s01];
                cpx v10 = state[s10];
                state[s01] = v01 * mat[0][1] + v10 * mat[1][0];
                state[s10] = v10 * mat[0][1] + v01 * mat[1][0];
            }
        } else {
            UNREACHABLE();
        }

        // apply error to gate
        for (auto target: targets) {
            int t1 = target;
            int t2 = target + n2;
            idx_t low_bit = t1;
            idx_t high_bit = t2;
            idx_t mask_inner = (idx_t(1) << low_bit) - 1;
            idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
            idx_t mask_outer = (idx_t(1) << (numQubits - 2)) - 1 - mask_inner - mask_middle;
            #pragma omp parallel for
            for (idx_t i = 0; i < (idx_t(1) << (numQubits - 2)); i++) {
                idx_t s00 = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                idx_t s01 = s00 | (idx_t(1) << t1);
                idx_t s10 = s00 | (idx_t(1) << t2);
                idx_t s11 = s01 | s10;
                cpx v00 = state[s00];
                cpx v01 = state[s01];
                cpx v10 = state[s10];
                cpx v11 = state[s11];
                cpx n00 = cpx(0.0);
                cpx n01 = cpx(0.0);
                cpx n10 = cpx(0.0);
                cpx n11 = cpx(0.0);
                for (auto& g: errs[target].gates) {
                    cpx mat[2][2] = {{g.mat[0][0], g.mat[0][1]}, {g.mat[1][0], g.mat[1][1]}};
                    printf("mat %f %f %f %f\n", mat[0][0].real(), mat[0][1].real(), mat[1][0].real(), mat[1][1].real());
                    cpx w00 = mat[0][0] * v00 + mat[0][1] * v10;
                    cpx w01 = mat[0][0] * v01 + mat[0][1] * v11;
                    cpx w10 = mat[1][0] * v00 + mat[1][1] * v10;
                    cpx w11 = mat[1][0] * v01 + mat[1][1] * v11;
                    mat[0][0] = std::conj(mat[0][0]);
                    mat[0][1] = std::conj(mat[0][1]);
                    mat[1][0] = std::conj(mat[1][0]);
                    mat[1][1] = std::conj(mat[1][1]);
                    n00 += w00 * mat[0][0] + w01 * mat[0][1];
                    n01 += w00 * mat[1][0] + w01 * mat[1][1];
                    n10 += w10 * mat[0][0] + w11 * mat[0][1];
                    n11 += w10 * mat[1][0] + w11 * mat[1][1];
                }
                state[s00] = n00;
                state[s01] = n01;
                state[s10] = n10;
                state[s11] = n11;
            }
        }
    }
    for (int i = 0; i < (1 << n2); i++) {
        for (int j = 0; j < (1 << n2); j++) {
            printf("%.3f,%.3f ", state[i << n2 | j].real(), state[i << n2 | j].imag());
        }
        printf("\n");
    }
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
#if MODE != 2
    for (int i = 0; i < numQubits; i++) {
        if (idx >> i & 1)
            id |= idx_t(1) << pos[i];
    }
#else
    for (int i = 0; i < numQubits / 2; i++) {
        idx_t msk = (idx >> (i * 2) & 3);
        id |= msk << (pos[i] * 2);
    }
#endif
    return id;
}

idx_t Circuit::toLogicID(idx_t idx) {
    idx_t id = 0;
    auto& pos = schedule.finalState.pos;
#if MODE != 2
    for (int i = 0; i < numQubits; i++) {
        if (idx >> pos[i] & 1)
            id |= idx_t(1) << i;
    }
#else
    for (int i = 0; i < numQubits / 2; i++) {
        idx_t msk = (idx >> (pos[i] * 2) & 3);
        id |= msk << (i * 2);
    }
#endif
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

void Circuit::duplicate_conj() {
    std::vector<Gate> duplicate_gates = gates;
    int nd2 = numQubits / 2;
    for (auto& gate: duplicate_gates) {
        gate.gateID = Gate::newID();
        gate.targetQubit += nd2;
        if (gate.isControlGate()) {
            gate.controlQubit +=nd2;
            if (gate.type == GateType::CY) {
                gate.type = GateType::CU;
                gate.name = "CU";
            }
        } else if (gate.isTwoQubitGate()) {
            gate.encodeQubit += nd2;
        } else if (gate.isSingleGate()) {
            if (gate.type == GateType::Y || gate.type == GateType::S || gate.type == GateType::SDG || gate.type == GateType::T || gate.type == GateType::TDG || gate.type == GateType::GII) {
                gate.type = GateType::U;
                gate.name = "U";
            }
        } else {
            UNIMPLEMENTED();
        }
        gate.mat[0][0] = std::conj(gate.mat[0][0]);
        gate.mat[0][1] = std::conj(gate.mat[0][1]);
        gate.mat[1][0] = std::conj(gate.mat[1][0]);
        gate.mat[1][1] = std::conj(gate.mat[1][1]);
        gates.push_back(gate);
    }
}

void Circuit::masterCompile() {
    Logger::add("Total Gates %d", int(gates.size()));
    this->transform();
#if GPU_BACKEND == 1 || GPU_BACKEND == 2 || GPU_BACKEND == 3 || GPU_BACKEND == 4 || GPU_BACKEND == 5
#if MODE == 2
    Compiler compiler(numQubits / 2, gates, MyGlobalVars::bit / 2);
#else
    Compiler compiler(numQubits, gates, MyGlobalVars::bit);
#endif
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
#if MODE == 2
    schedule.dump(numQubits / 2);
#else
    schedule.dump(numQubits);
#endif
#endif
#else
    schedule.finalState = State(numQubits);
#endif
}

void Circuit::compile() {
    auto start = chrono::system_clock::now();
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
            collected[i].print(numQubits);
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
        idx_t idx;
        switch (MODE) {
            case 0: { idx = i; break; }
            case 1: { idx = ((1 << (numQubits / 2)) + 1) * i; break; }
            case 2: { idx = duplicate_bit(i); break; }
        }
        if (localAmpAt(idx, item)) {
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
        idx_t idx;
        switch (MODE) {
            case 0: { idx = i; break; }
            case 1: { idx = ((1 << (numQubits / 2)) + 1) * i; break; }
            case 2: { idx = duplicate_bit(i); break; }
        }
        results.push_back(ampAt(idx));
    }
#ifdef SHOW_SCHEDULE
    for (int i = 0; i < numQubits; i++) {
        results.push_back(ampAt(1ll << i));
    }
    results.push_back(ampAt((1ll << numQubits) - 1));
#endif
    for (auto& item: results)
        item.print(numQubits);
    results.clear();
    for (idx_t i = 0; i < (1ll << numQubits); i++) {
        if (std::norm(result[i]) > 0.001) {
            idx_t logicID = toLogicID(i);
            if (MODE == 0) {
                if (logicID >= 128) {
                    results.push_back(ResultItem(toLogicID(i), result[i]));
                }
            } else {
                if ((logicID & 0x5555555555555555ll) != (logicID >> 1 & 0x5555555555555555ll) || logicID >= 128 * 128) {
                    results.push_back(ResultItem(toLogicID(i), result[i]));
                }
            }
        }
    }
    sort(results.begin(), results.end());
    for (auto& item: results)
        item.print(numQubits);
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

#if MODE == 2
void single_error_fusion(std::vector<Gate> &gates, int numQubits, bool erased[]) {
    for (int i = 0; i < (int) gates.size(); i++) {
        if (erased[i]) continue;
        if (gates[i].isSingleGate()) {
            Gate cpy = gates[i];
            gates[i] = Gate::ID(gates[i].targetQubit);
            gates[i].gateID = cpy.gateID;
            gates[i].controlErrors = cpy.controlErrors;
            assert(gates[i].controlErrors.size() == 0);
            gates[i].targetErrors = cpy.targetErrors;
            for (auto& err: gates[i].targetErrors) {
                cpx mat00 = err.mat00 * std::conj(cpy.mat[0][0]) + err.mat01 * std::conj(cpy.mat[1][0]);
                cpx mat01 = err.mat00 * std::conj(cpy.mat[0][1]) + err.mat01 * std::conj(cpy.mat[1][1]);
                cpx mat10 = err.mat10 * std::conj(cpy.mat[0][0]) + err.mat11 * std::conj(cpy.mat[1][0]);
                cpx mat11 = err.mat10 * std::conj(cpy.mat[0][1]) + err.mat11 * std::conj(cpy.mat[1][1]);
                err.type = GateType::U;
                err.mat00 = mat00; err.mat01 = mat01; err.mat10 = mat10; err.mat11 = mat11;
            }
        }
    }
}
#endif

inline bool eps_equal(cpx a, cpx b) {
    const value_t eps = 1e-14;
    return a.real() - b.real() < eps && b.real() - a.real() < eps && a.imag() - b.imag() < eps && b.imag() - a.imag() < eps;
}

void Circuit::transform() {
    bool *erased = new bool[gates.size()];
    memset(erased, 0, sizeof(bool) * gates.size());
#if MODE != 2
    hczh2cx(this->gates, numQubits, erased);
    single_qubit_fusion(this->gates, numQubits, erased);
#else
    single_error_fusion(this->gates, numQubits, erased);
#endif
    std::vector<Gate> new_gates;
    for (int i = 0; i < (int) gates.size(); i++)
        if (!erased[i])
            new_gates.push_back(gates[i]);
    gates = new_gates;
    delete[] erased;
}
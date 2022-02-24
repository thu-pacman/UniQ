#include "cpu_executor.h"
#include <omp.h>
#include <cstring>
#include <assert.h>
#include <x86intrin.h>

namespace CpuImpl {

CpuExecutor::CpuExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): Executor(deviceStateVec, numQubits, schedule) {}

void CpuExecutor::transpose(std::vector<std::shared_ptr<hptt::Transpose<cpx>>> plans) {
    plans[0]->setInputPtr(deviceStateVec[0]);
    plans[0]->setOutputPtr(deviceBuffer[0]);
    plans[0]->execute();
}

void CpuExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit;
    idx_t numElements = 1ll << numLocalQubit;
    int numPart = numSlice / commSize;
    idx_t partSize = numElements / numSlice;
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    int sliceID = 0;
    for (int xr = 0; xr < commSize; xr++) {
        for (int p = 0; p < numPart; p++) {
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                int comm_a = comm[a] % MyGlobalVars::localGPUs;
                int srcPart = a % commSize * numPart + p;
                int dstPart = b % commSize * numPart + p;
#if USE_MPI
                if (a == b) {
                    memcpy(
                        deviceStateVec[comm_a] + dstPart * partSize,
                        deviceBuffer[comm_a] + srcPart * partSize,
                        partSize * sizeof(cpx)
                    );
                } else {
                    checkMPIErrors(MPI_Sendrecv(
                        deviceBuffer[comm_a] + dstPart * partSize, partSize, MPI_Complex, comm[b], 0,
                        deviceStateVec[comm_a] + dstPart * partSize, partSize, MPI_Complex, comm[b], MPI_ANY_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE
                    ));
                }
#else
                UNIMPLEMENTED();
#endif
                partID[sliceID * MyGlobalVars::localGPUs + comm_a] = dstPart;
                peer[sliceID * MyGlobalVars::localGPUs + comm_a] = comm[b];
            }
            // events should be recorded after ncclGroupEnd
#ifdef ENABLE_OVERLAP
            
            UNIMPLEMENTED();
#endif
            sliceID++;
        }
    }
#ifndef ENABLE_OVERLAP
    this->eventBarrierAll();
#endif
}

#define FOLLOW_NEXT(TYPE) \
case GateType::TYPE: // no break

#if GPU_BACKEND==1

inline void fetch_data(cpx* local, const cpx* deviceStateVec, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    for (x = ((1 << LOCAL_QUBIT_SIZE) - 1), y = relatedQubits; x >= 0; x--, y = relatedQubits & (y-1)) {
        local[x] = deviceStateVec[bias | y];
    }
}

inline void save_data(cpx* deviceStateVec, const cpx* local, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    for (x = ((1 << LOCAL_QUBIT_SIZE) - 1), y = relatedQubits; x >= 0; x--, y = relatedQubits & (y-1)) {
        deviceStateVec[bias | y] = local[x];
    }
}

inline void apply_gate_group(cpx* local, int numGates, int blockID, KernelGate hostGates[]) {
    for (int i = 0; i < numGates; i++) {
        auto& gate = hostGates[i];
        int controlQubit = gate.controlQubit;
        int targetQubit = gate.targetQubit;
        char controlIsGlobal = gate.controlIsGlobal;
        char targetIsGlobal = gate.targetIsGlobal;
        if (!controlIsGlobal) {
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                int low_bit = std::min(controlQubit, targetQubit);
                int high_bit = std::max(controlQubit, targetQubit);
                // TODO: switch (gate)
                #ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 2)) - (1 << low_bit));
                __m256i mask_outer = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << high_bit));
                __m256i ctr_flag = _mm256_set1_epi32(1 << controlQubit);
                __m256i tar_flag = _mm256_set1_epi32(1 << targetQubit);
                assert(m % 8 == 0);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                for (int j = 0; j < m; j += 8) {
                    __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                    lo = _mm256_add_epi32(lo, _mm256_and_si256(lo, mask_outer));
                    lo = _mm256_add_epi32(lo, ctr_flag);
                    __m256i hi = _mm256_add_epi32(lo, tar_flag);
                    lo = _mm256_add_epi32(lo, lo);
                    hi = _mm256_add_epi32(hi, hi);
                    __m512d lo_real = _mm512_i32gather_pd(lo, local, 8);
                    __m512d lo_imag = _mm512_i32gather_pd(lo, reinterpret_cast<value_t*>(local) + 1, 8);
                    __m512d hi_real = _mm512_i32gather_pd(hi, local, 8);
                    __m512d hi_imag = _mm512_i32gather_pd(hi, reinterpret_cast<value_t*>(local) + 1, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = _mm512_set1_pd(gate.i00);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = _mm512_set1_pd(gate.i01);
                    __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                    lo_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_fmadd_pd(hi_real, r01, lo_real_new));
                    __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                    lo_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_fmadd_pd(hi_real, i01, lo_imag_new));
                    _mm512_i32scatter_pd(local, lo, lo_real_new, 8);
                    _mm512_i32scatter_pd(reinterpret_cast<value_t*>(local) + 1, lo, lo_imag_new, 8);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = _mm512_set1_pd(gate.i10);
                    __m512d hi_real_new = _mm512_fnmadd_pd(lo_imag, i10, _mm512_mul_pd(lo_real, r10));
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = _mm512_set1_pd(gate.i11);
                    hi_real_new = _mm512_fnmadd_pd(hi_imag, i11, _mm512_fmadd_pd(hi_real, r11, hi_real_new));
                    __m512d hi_imag_new = _mm512_fmadd_pd(lo_imag, r10, _mm512_mul_pd(lo_real, i10));
                    hi_imag_new = _mm512_fmadd_pd(hi_imag, r11, _mm512_fmadd_pd(hi_real, i11, hi_imag_new));
                    _mm512_i32scatter_pd(local, hi, hi_real_new, 8);
                    _mm512_i32scatter_pd(reinterpret_cast<value_t*>(local) + 1, hi, hi_imag_new, 8);
                    idx = _mm256_add_epi32(idx, inc);
                }
                #else
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 2)) - (1 << low_bit);
                int mask_outer = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << high_bit);
                #pragma ivdep
                for (int j = 0; j < m; j++) {
                    int lo = j + (j & mask_inner);
                    lo = lo + (lo & mask_outer);
                    lo += 1 << controlQubit;
                    int hi = lo | (1 << targetQubit);
                    cpx lo_val = local[lo];
                    cpx hi_val = local[hi];
                    local[lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local[hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                #endif
            } else {
                assert(hostGates[i].type == GateType::CZ || hostGates[i].type == GateType::CU1 || hostGates[i].type == GateType::CRZ);
                bool isHighBlock = (blockID >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << controlQubit);
                if (!isHighBlock){
                    if (hostGates[i].type == GateType::CRZ) {
                        for (int j = 0; j < m; j++) {
                            int x = j + (j & mask_inner) + (1 << controlQubit);
                            local[x] = local[x] * cpx(gate.r00, gate.i00);
                        }
                    }
                } else {
                    for (int j = 0; j < m; j++) {
                        int x = j + (j & mask_inner) + (1 << controlQubit);
                        local[x] = local[x] * cpx(gate.r11, gate.i11);
                    }
                }
            }
        } else {
            if (controlIsGlobal == 1 && !((blockID>> controlQubit) & 1)) {
                continue;
            }
            if (!targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                #ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << gate.targetQubit));
                __m256i tar_flag = _mm256_set1_epi32(1 << targetQubit);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                for (int j = 0; j < m; j += 8) {
                    __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                    __m256i hi = _mm256_add_epi32(lo, tar_flag);
                    lo = _mm256_add_epi32(lo, lo);
                    hi = _mm256_add_epi32(hi, hi);
                    __m512d lo_real = _mm512_i32gather_pd(lo, local, 8);
                    __m512d lo_imag = _mm512_i32gather_pd(lo, reinterpret_cast<value_t*>(local) + 1, 8);
                    __m512d hi_real = _mm512_i32gather_pd(hi, local, 8);
                    __m512d hi_imag = _mm512_i32gather_pd(hi, reinterpret_cast<value_t*>(local) + 1, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = _mm512_set1_pd(gate.i00);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = _mm512_set1_pd(gate.i01);
                    __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                    lo_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_fmadd_pd(hi_real, r01, lo_real_new));
                    __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                    lo_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_fmadd_pd(hi_real, i01, lo_imag_new));
                    _mm512_i32scatter_pd(local, lo, lo_real_new, 8);
                    _mm512_i32scatter_pd(reinterpret_cast<value_t*>(local) + 1, lo, lo_imag_new, 8);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = _mm512_set1_pd(gate.i10);
                    __m512d hi_real_new = _mm512_fnmadd_pd(lo_imag, i10, _mm512_mul_pd(lo_real, r10));
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = _mm512_set1_pd(gate.i11);
                    hi_real_new = _mm512_fnmadd_pd(hi_imag, i11, _mm512_fmadd_pd(hi_real, r11, hi_real_new));
                    __m512d hi_imag_new = _mm512_fmadd_pd(lo_imag, r10, _mm512_mul_pd(lo_real, i10));
                    hi_imag_new = _mm512_fmadd_pd(hi_imag, r11, _mm512_fmadd_pd(hi_real, i11, hi_imag_new));
                    _mm512_i32scatter_pd(local, hi, hi_real_new, 8);
                    _mm512_i32scatter_pd(reinterpret_cast<value_t*>(local) + 1, hi, hi_imag_new, 8);
                    idx = _mm256_add_epi32(idx, inc);
                }
                #else
                // TODO: switch (gate)
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << gate.targetQubit);
                #pragma ivdep
                for (int j = 0; j < m; j++) {
                    int lo = j + (j & mask_inner);
                    int hi = lo | (1 << targetQubit);
                    cpx lo_val = local[lo];
                    cpx hi_val = local[hi];
                    local[lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local[hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                #endif
            } else {
                bool isHighBlock = (blockID >> targetQubit) & 1;
                // TODO: switch (gate)
                int m = 1 << LOCAL_QUBIT_SIZE;
                if (!isHighBlock){
                    #pragma ivdep
                    for (int j = 0; j < m; j++) {
                        local[j] = local[j] * cpx(gate.r00, gate.i00);
                    }
                } else {
                    #pragma ivdep
                    for (int j = 0; j < m; j++) {
                        local[j] = local[j] * cpx(gate.r11, gate.i11);
                    }
                }
            }
        }
    }
}

void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    cpx* sv = deviceStateVec[0];
    #pragma omp parallel for
    for (int blockID = 0; blockID < (1 << (numLocalQubits - LOCAL_QUBIT_SIZE)); blockID++) {
        cpx local[1 << LOCAL_QUBIT_SIZE];
        idx_t blockHot = (idx_t(1) << numLocalQubits) - 1 - relatedQubits;
        unsigned int bias = 0;
        {
            int bid = blockID;
            for (unsigned int bit = 1; bit < (1u << numLocalQubits); bit <<= 1) {
                if (blockHot & bit) {
                    if (bid & 1)
                        bias |= bit;
                    bid >>= 1;
                }
            }
        }
        fetch_data(local, sv, bias, relatedQubits);
        apply_gate_group(local, gates.size(), blockID, hostGates);
        save_data(sv, local, bias, relatedQubits);
    }
}
#elif GPU_BACKEND==2
void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    #pragma omp parallel
    for (int i = 0; i < int(gates.size()); i++) {
        auto& gate = hostGates[i];
        switch (gate.type) {
            FOLLOW_NEXT(CNOT)
            FOLLOW_NEXT(CY)
            FOLLOW_NEXT(CZ)
            FOLLOW_NEXT(CRX)
            FOLLOW_NEXT(CRY)
            FOLLOW_NEXT(CU1)
            FOLLOW_NEXT(CRZ)
            case GateType::CU: {
                int c = gate.controlQubit;
                int t = gate.targetQubit;
                idx_t low_bit = std::min(c, t);
                idx_t high_bit = std::max(c, t);
                idx_t mask_inner = (idx_t(1) << low_bit) - 1;
                idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
                idx_t mask_outer = (idx_t(1) << (numLocalQubits - 2)) - 1 - mask_inner - mask_middle;
                #pragma omp for
                for (idx_t i = 0; i < (idx_t(1) << (numLocalQubits - 2)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                    lo |= idx_t(1) << c;
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    deviceStateVec[0][hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                break;
            }
            FOLLOW_NEXT(U1)
            FOLLOW_NEXT(U2)
            FOLLOW_NEXT(U3)
            FOLLOW_NEXT(U)
            FOLLOW_NEXT(H)
            FOLLOW_NEXT(X)
            FOLLOW_NEXT(Y)
            FOLLOW_NEXT(Z)
            FOLLOW_NEXT(S)
            FOLLOW_NEXT(SDG)
            FOLLOW_NEXT(T)
            FOLLOW_NEXT(TDG)
            FOLLOW_NEXT(RX)
            FOLLOW_NEXT(RY)
            FOLLOW_NEXT(RZ)
            FOLLOW_NEXT(ID)
            FOLLOW_NEXT(GII)
            FOLLOW_NEXT(GZZ)
            FOLLOW_NEXT(GOC)
            case GateType::GCC: {
                int t = gate.targetQubit;
                idx_t mask_inner = (idx_t(1) << t) - 1;
                idx_t mask_outer = (idx_t(1) << (numLocalQubits - 1)) - 1 - mask_inner;
                #pragma omp for
                for (idx_t i = 0; i < (idx_t(1) << (numLocalQubits - 1)); i++) {
                    idx_t lo = (i & mask_inner) + ((i & mask_outer) << 1);
                    idx_t hi = lo | (idx_t(1) << t);
                    cpx lo_val = deviceStateVec[0][lo];
                    cpx hi_val = deviceStateVec[0][hi];
                    deviceStateVec[0][lo] = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    deviceStateVec[0][hi] = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                }
                break;
            }
            default: {
                UNIMPLEMENTED();
            }
            #pragma omp barrier
        }
    }
}
#else
void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    UNIMPLEMENTED()
}
#endif

void CpuExecutor::deviceFinalize() {}

void CpuExecutor::allBarrier() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}


void CpuExecutor::dm_transpose()  { UNIMPLEMENTED(); }
void CpuExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) { UNIMPLEMENTED(); }
void CpuExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrierAll() { checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD)); }
}
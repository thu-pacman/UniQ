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
    partID.resize(numSlice * MyGlobalVars::localGPUs);
    peer.resize(numSlice * MyGlobalVars::localGPUs);
    int sliceID = 0;
#ifdef ALL_TO_ALL
    idx_t partSize = numElements / commSize;
    int newRank = -1;
    for (int i = 0; i < MyGlobalVars::numGPUs; i++) {
        if (comm[i] == MyMPI::rank) {
            newRank = i;
            break;
        }
    }
    MPI_Group world_group, new_group;
    checkMPIErrors(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
    int ranks[commSize];
    for (int i = 0; i < commSize; i++)
        ranks[i] = (newRank - newRank % commSize) + i;
    checkMPIErrors(MPI_Group_incl(world_group, commSize, ranks, &new_group));
    MPI_Comm new_communicator;
    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_communicator);

    checkMPIErrors(MPI_Alltoall(
        deviceBuffer[0], partSize, MPI_Complex,
        deviceStateVec[0], partSize, MPI_Complex,
        new_communicator
    ))

#else
    idx_t partSize = numElements / numSlice;
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
#endif
#ifndef ENABLE_OVERLAP
    this->eventBarrierAll();
#endif
}

#define FOLLOW_NEXT(TYPE) \
case GateType::TYPE: // no break

#if GPU_BACKEND==1

inline void fetch_data(value_t* local_real, value_t* local_imag, const cpx* deviceStateVec, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    idx_t mask = (1 << COALESCE_GLOBAL) - 1;
    assert((relatedQubits & mask) == mask);
    relatedQubits -= mask;
    for (x = (1 << LOCAL_QUBIT_SIZE) - 1 - mask, y = relatedQubits; x >= 0; x -= (1 << COALESCE_GLOBAL), y = relatedQubits & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << COALESCE_GLOBAL); i++) {
            local_real[x + i] = deviceStateVec[(bias | y) + i].real();
            local_imag[x + i] = deviceStateVec[(bias | y) + i].imag();
            // printf("fetch %d <- %d\n", x + i, (bias | y) + i);
        }
    }
}

inline void save_data(cpx* deviceStateVec, const value_t* local_real, const value_t* local_imag, int bias, idx_t relatedQubits) {
    int x;
    unsigned int y;
    idx_t mask = (1 << COALESCE_GLOBAL) - 1;
    assert((relatedQubits & mask) == mask);
    relatedQubits -= mask;
    for (x = (1 << LOCAL_QUBIT_SIZE) - 1 - mask, y = relatedQubits; x >= 0; x -= (1 << COALESCE_GLOBAL), y = relatedQubits & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << COALESCE_GLOBAL); i++) {
            deviceStateVec[(bias | y) + i].real(local_real[x + i]);
            deviceStateVec[(bias | y) + i].imag(local_imag[x + i]);
        }
    }
}

inline void apply_gate_group(value_t* local_real, value_t* local_imag, int numGates, int blockID, KernelGate hostGates[]) {
    for (int i = 0; i < numGates; i++) {
        auto& gate = hostGates[i];
        int controlQubit = gate.controlQubit;
        int targetQubit = gate.targetQubit;
        char controlIsGlobal = gate.controlIsGlobal;
        char targetIsGlobal = gate.targetIsGlobal;
        if (controlQubit == -2) { // mcGate
            UNIMPLEMENTED();
        } else if (controlQubit == -3) {
            if (!controlIsGlobal && !targetIsGlobal) {
                int m = 1 << (LOCAL_QUBIT_SIZE - 2);
                int low_bit = std::min((int) gate.encodeQubit, targetQubit);
                int high_bit = std::max((int) gate.encodeQubit, targetQubit);
                #ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 2)) - (1 << low_bit));
                __m256i mask_outer = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << high_bit));
                __m256i ctr_flag = _mm256_set1_epi32(1 << gate.encodeQubit);
                __m256i tar_flag = _mm256_set1_epi32(1 << targetQubit);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                for (int j = 0; j < m; j += 8) {
                    __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                    lo = _mm256_add_epi32(lo, _mm256_and_si256(lo, mask_outer));
                    __m256i s00 = lo;
                    __m256i s01 = _mm256_add_epi32(s00, tar_flag);
                    __m256i s10 = _mm256_add_epi32(s00, ctr_flag);
                    __m256i s11 = _mm256_or_si256(s01, s10);

                    __m512d v00_real = _mm512_i32gather_pd(s00, local_real, 8);
                    __m512d v00_imag = _mm512_i32gather_pd(s00, local_imag, 8);
                    __m512d v11_real = _mm512_i32gather_pd(s11, local_real, 8);
                    __m512d v11_imag = _mm512_i32gather_pd(s11, local_imag, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = _mm512_set1_pd(gate.i00);
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = _mm512_set1_pd(gate.i11);
                    __m512d v00_real_new = _mm512_fnmadd_pd(v00_imag, i00, _mm512_mul_pd(v00_real, r00));
                    v00_real_new = _mm512_fnmadd_pd(v11_imag, i11, _mm512_fmadd_pd(v11_real, r11, v00_real_new));
                    __m512d v00_imag_new = _mm512_fmadd_pd(v00_imag, r00, _mm512_mul_pd(v00_real, i00));
                    v00_imag_new = _mm512_fmadd_pd(v11_imag, r11, _mm512_fmadd_pd(v11_real, i11, v00_imag_new));
                    _mm512_i32scatter_pd(local_real, s00, v00_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, s00, v00_imag_new, 8);
                    __m512d v11_real_new = _mm512_fnmadd_pd(v11_imag, i00, _mm512_mul_pd(v11_real, r00));
                    v11_real_new = _mm512_fnmadd_pd(v00_imag, i11, _mm512_fmadd_pd(v00_real, r11, v11_real_new));
                    __m512d v11_imag_new = _mm512_fmadd_pd(v11_imag, r00, _mm512_mul_pd(v11_real, i00));
                    v11_imag_new = _mm512_fmadd_pd(v00_imag, r11, _mm512_fmadd_pd(v00_real, i11, v11_imag_new));
                    _mm512_i32scatter_pd(local_real, s11, v11_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, s11, v11_imag_new, 8);

                    __m512d v01_real = _mm512_i32gather_pd(s01, local_real, 8);
                    __m512d v01_imag = _mm512_i32gather_pd(s01, local_imag, 8);
                    __m512d v10_real = _mm512_i32gather_pd(s10, local_real, 8);
                    __m512d v10_imag = _mm512_i32gather_pd(s10, local_imag, 8);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = _mm512_set1_pd(gate.i01);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = _mm512_set1_pd(gate.i10);
                    __m512d v01_real_new = _mm512_fnmadd_pd(v01_imag, i01, _mm512_mul_pd(v01_real, r01));
                    v01_real_new = _mm512_fnmadd_pd(v10_imag, i10, _mm512_fmadd_pd(v10_real, r10, v01_real_new));
                    __m512d v01_imag_new = _mm512_fmadd_pd(v01_imag, r01, _mm512_mul_pd(v01_real, i01));
                    v01_imag_new = _mm512_fmadd_pd(v10_imag, r10, _mm512_fmadd_pd(v10_real, i10, v01_imag_new));
                    _mm512_i32scatter_pd(local_real, s01, v01_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, s01, v01_imag_new, 8);
                    __m512d v10_real_new = _mm512_fnmadd_pd(v10_imag, i01, _mm512_mul_pd(v10_real, r01));
                    v10_real_new = _mm512_fnmadd_pd(v01_imag, i10, _mm512_fmadd_pd(v01_real, r10, v10_real_new));
                    __m512d v10_imag_new = _mm512_fmadd_pd(v10_imag, r01, _mm512_mul_pd(v10_real, i01));
                    v10_imag_new = _mm512_fmadd_pd(v01_imag, r10, _mm512_fmadd_pd(v01_real, i10, v10_imag_new));
                    _mm512_i32scatter_pd(local_real, s10, v10_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, s10, v10_imag_new, 8);

                    idx = _mm256_add_epi32(idx, inc);
                }
                #else
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 2)) - (1 << low_bit);
                int mask_outer = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << high_bit);
                for (int j = 0; j < m; j++) {
                    int lo = j + (j & mask_inner);
                    lo = lo + (lo & mask_outer);
                    int s00 = lo;
                    int s01 = s00 | 1 << targetQubit;
                    int s10 = s00 | 1 << gate.encodeQubit;
                    int s11 = s01 | s10;
                    cpx v00 = cpx(local_real[s00], local_imag[s00]);
                    cpx v11 = cpx(local_real[s11], local_imag[s11]);
                    cpx n00 = v00 * cpx(gate.r00, gate.i00) + v11 * cpx(gate.r11, gate.i11);
                    cpx n11 = v11 * cpx(gate.r00, gate.i00) + v00 * cpx(gate.r11, gate.i11);
                    local_real[s00] = n00.real();
                    local_imag[s00] = n00.imag();
                    local_real[s11] = n11.real();
                    local_imag[s11] = n11.imag();
                    cpx v01 = cpx(local_real[s01], local_imag[s01]);
                    cpx v10 = cpx(local_real[s10], local_imag[s10]);
                    cpx n01 = v01 * cpx(gate.r01, gate.i01) + v10 * cpx(gate.r10, gate.i10);
                    cpx n10 = v10 * cpx(gate.r01, gate.i01) + v01 * cpx(gate.r10, gate.i10);
                    local_real[s01] = n01.real();
                    local_imag[s01] = n01.imag();
                    local_real[s10] = n10.real();
                    local_imag[s10] = n10.imag();
                }
                #endif
            } else if (controlIsGlobal && !targetIsGlobal) {
                bool isHighBlock = (blockID >> gate.encodeQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                #ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << gate.targetQubit));
                __m256i tar_flag = _mm256_set1_epi32(1 << targetQubit);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                if (!isHighBlock){
                    for (int j = 0; j < m; j += 8) {
                        __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                        __m512d lo_real = _mm512_i32gather_pd(lo, local_real, 8);
                        __m512d lo_imag = _mm512_i32gather_pd(lo, local_imag, 8);
                        __m512d r00 = _mm512_set1_pd(gate.r00);
                        __m512d i00 = _mm512_set1_pd(gate.i00);
                        __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                        _mm512_i32scatter_pd(local_real, lo, lo_real_new, 8);
                        __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                        _mm512_i32scatter_pd(local_imag, lo, lo_imag_new, 8);

                        __m256i hi = _mm256_add_epi32(lo, tar_flag);
                        __m512d hi_real = _mm512_i32gather_pd(hi, local_real, 8);
                        __m512d hi_imag = _mm512_i32gather_pd(hi, local_imag, 8);
                        __m512d r01 = _mm512_set1_pd(gate.r01);
                        __m512d i01 = _mm512_set1_pd(gate.i01);
                        __m512d hi_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_mul_pd(hi_real, r01));
                        _mm512_i32scatter_pd(local_real, hi, hi_real_new, 8);
                        __m512d hi_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_mul_pd(hi_real, i01));
                        _mm512_i32scatter_pd(local_imag, hi, hi_imag_new, 8);

                        idx = _mm256_add_epi32(idx, inc);
                    }
                } else {
                    for (int j = 0; j < m; j += 8) {
                        __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                        __m512d lo_real = _mm512_i32gather_pd(lo, local_real, 8);
                        __m512d lo_imag = _mm512_i32gather_pd(lo, local_imag, 8);
                        __m512d r01 = _mm512_set1_pd(gate.r01);
                        __m512d i01 = _mm512_set1_pd(gate.i01);
                        __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i01, _mm512_mul_pd(lo_real, r01));
                        _mm512_i32scatter_pd(local_real, lo, lo_real_new, 8);
                        __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r01, _mm512_mul_pd(lo_real, i01));
                        _mm512_i32scatter_pd(local_imag, lo, lo_imag_new, 8);

                        __m256i hi = _mm256_add_epi32(lo, tar_flag);
                        __m512d hi_real = _mm512_i32gather_pd(hi, local_real, 8);
                        __m512d hi_imag = _mm512_i32gather_pd(hi, local_imag, 8);
                        __m512d r00 = _mm512_set1_pd(gate.r00);
                        __m512d i00 = _mm512_set1_pd(gate.i00);
                        __m512d hi_real_new = _mm512_fnmadd_pd(hi_imag, i00, _mm512_mul_pd(hi_real, r00));
                        _mm512_i32scatter_pd(local_real, hi, hi_real_new, 8);
                        __m512d hi_imag_new = _mm512_fmadd_pd(hi_imag, r00, _mm512_mul_pd(hi_real, i00));
                        _mm512_i32scatter_pd(local_imag, hi, hi_imag_new, 8);

                        idx = _mm256_add_epi32(idx, inc);

                    }
                }
                #else
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << targetQubit);
                if (!isHighBlock){
                    #pragma ivdep
                    for (int j = 0; j < m; j++) {
                        int lo = j + (j & mask_inner);
                        int hi = lo + (1 << targetQubit);
                        cpx new_lo = cpx(local_real[lo], local_imag[lo]) * cpx(gate.r00, gate.i00);
                        local_real[lo] = new_lo.real();
                        local_imag[lo] = new_lo.imag();
                        cpx new_hi = cpx(local_real[hi], local_imag[hi]) * cpx(gate.r01, gate.i01);
                        local_real[hi] = new_hi.real();
                        local_imag[hi] = new_hi.imag();
                    }
                } else {
                    #pragma ivdep
                    for (int j = 0; j < m; j++) {
                        int lo = j + (j & mask_inner);
                        int hi = lo + (1 << targetQubit);
                        cpx new_lo = cpx(local_real[lo], local_imag[lo]) * cpx(gate.r01, gate.i01);
                        local_real[lo] = new_lo.real();
                        local_imag[lo] = new_lo.imag();
                        cpx new_hi = cpx(local_real[hi], local_imag[hi]) * cpx(gate.r00, gate.i00);
                        local_real[hi] = new_hi.real();
                        local_imag[hi] = new_hi.imag();
                    }
                }
                #endif
            } else {
                UNIMPLEMENTED();
            }
        } else if (!controlIsGlobal) {
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
                    // lo = _mm256_add_epi32(lo, lo);
                    // hi = _mm256_add_epi32(hi, hi);
                    __m512d lo_real = _mm512_i32gather_pd(lo, local_real, 8);
                    __m512d lo_imag = _mm512_i32gather_pd(lo, local_imag, 8);
                    __m512d hi_real = _mm512_i32gather_pd(hi, local_real, 8);
                    __m512d hi_imag = _mm512_i32gather_pd(hi, local_imag, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = _mm512_set1_pd(gate.i00);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = _mm512_set1_pd(gate.i01);
                    __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                    lo_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_fmadd_pd(hi_real, r01, lo_real_new));
                    __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                    lo_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_fmadd_pd(hi_real, i01, lo_imag_new));
                    _mm512_i32scatter_pd(local_real, lo, lo_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, lo, lo_imag_new, 8);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = _mm512_set1_pd(gate.i10);
                    __m512d hi_real_new = _mm512_fnmadd_pd(lo_imag, i10, _mm512_mul_pd(lo_real, r10));
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = _mm512_set1_pd(gate.i11);
                    hi_real_new = _mm512_fnmadd_pd(hi_imag, i11, _mm512_fmadd_pd(hi_real, r11, hi_real_new));
                    __m512d hi_imag_new = _mm512_fmadd_pd(lo_imag, r10, _mm512_mul_pd(lo_real, i10));
                    hi_imag_new = _mm512_fmadd_pd(hi_imag, r11, _mm512_fmadd_pd(hi_real, i11, hi_imag_new));
                    _mm512_i32scatter_pd(local_real, hi, hi_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, hi, hi_imag_new, 8);
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
                    cpx lo_val = cpx(local_real[lo], local_imag[lo]);
                    cpx hi_val = cpx(local_real[hi], local_imag[hi]);
                    cpx lo_val_new = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local_real[lo] = lo_val_new.real();
                    local_imag[lo] = lo_val_new.imag();
                    cpx hi_val_new = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                    local_real[hi] = hi_val_new.real();
                    local_imag[hi] = hi_val_new.imag();
                }
                #endif
            } else {
                assert(hostGates[i].type == GateType::CZ || hostGates[i].type == GateType::CU1 || hostGates[i].type == GateType::CRZ);
                bool isHighBlock = (blockID >> targetQubit) & 1;
                int m = 1 << (LOCAL_QUBIT_SIZE - 1);
                if (!isHighBlock){
                    if (hostGates[i].type == GateType::CRZ) {
                        #ifdef USE_AVX512
                        __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << controlQubit));
                        __m256i ctr_flag = _mm256_set1_epi32(1 << controlQubit);
                        __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        const __m256i inc = _mm256_set1_epi32(8);
                        for (int j = 0; j < m; j += 8) {
                            __m256i x = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                            x = _mm256_add_epi32(x, ctr_flag);
                            __m512d r00 = _mm512_set1_pd(gate.r00);
                            __m512d i00 = _mm512_set1_pd(gate.i00);
                            __m512d x_real = _mm512_i32gather_pd(x, local_real, 8);
                            __m512d x_imag = _mm512_i32gather_pd(x, local_imag, 8);
                            __m512d x_real_new = _mm512_fnmadd_pd(x_imag, i00, _mm512_mul_pd(x_real, r00));
                            __m512d x_imag_new = _mm512_fmadd_pd(x_imag, r00, _mm512_mul_pd(x_real, i00));
                            _mm512_i32scatter_pd(local_real, x, x_real_new, 8);
                            _mm512_i32scatter_pd(local_imag, x, x_imag_new, 8);
                            idx = _mm256_add_epi32(idx, inc);
                        }
                        #else
                        int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << controlQubit);
                        #pragma ivdep
                        for (int j = 0; j < m; j++) {
                            int x = j + (j & mask_inner) + (1 << controlQubit);
                            cpx new_val = cpx(local_real[x], local_imag[x]) * cpx(gate.r00, gate.i00);
                            local_real[x] = new_val.real();
                            local_imag[x] = new_val.imag();
                        }
                        #endif
                    }
                } else {
                    #ifdef USE_AVX512
                    __m256i mask_inner = _mm256_set1_epi32((1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << controlQubit));
                    __m256i ctr_flag = _mm256_set1_epi32(1 << controlQubit);
                    __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    const __m256i inc = _mm256_set1_epi32(8);
                    for (int j = 0; j < m; j += 8) {
                        __m256i x = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                        x = _mm256_add_epi32(x, ctr_flag);
                        __m512d r11 = _mm512_set1_pd(gate.r11);
                        __m512d i11 = _mm512_set1_pd(gate.i11);
                        __m512d x_real = _mm512_i32gather_pd(x, local_real, 8);
                        __m512d x_imag = _mm512_i32gather_pd(x, local_imag, 8);
                        __m512d x_real_new = _mm512_fnmadd_pd(x_imag, i11, _mm512_mul_pd(x_real, r11));
                        __m512d x_imag_new = _mm512_fmadd_pd(x_imag, r11, _mm512_mul_pd(x_real, i11));
                        _mm512_i32scatter_pd(local_real, x, x_real_new, 8);
                        _mm512_i32scatter_pd(local_imag, x, x_imag_new, 8);
                        idx = _mm256_add_epi32(idx, inc);
                    }
                    #else
                    int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << controlQubit);
                    #pragma ivdep
                    for (int j = 0; j < m; j++) {
                        int x = j + (j & mask_inner) + (1 << controlQubit);
                        cpx new_val = cpx(local_real[x], local_imag[x]) * cpx(gate.r11, gate.i11);
                        local_real[x] = new_val.real();
                        local_imag[x] = new_val.imag();
                    }
                    #endif
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
                    __m512d lo_real = _mm512_i32gather_pd(lo, local_real, 8);
                    __m512d lo_imag = _mm512_i32gather_pd(lo, local_imag, 8);
                    __m512d hi_real = _mm512_i32gather_pd(hi, local_real, 8);
                    __m512d hi_imag = _mm512_i32gather_pd(hi, local_imag, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = _mm512_set1_pd(gate.i00);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = _mm512_set1_pd(gate.i01);
                    __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                    lo_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_fmadd_pd(hi_real, r01, lo_real_new));
                    __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                    lo_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_fmadd_pd(hi_real, i01, lo_imag_new));
                    _mm512_i32scatter_pd(local_real, lo, lo_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, lo, lo_imag_new, 8);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = _mm512_set1_pd(gate.i10);
                    __m512d hi_real_new = _mm512_fnmadd_pd(lo_imag, i10, _mm512_mul_pd(lo_real, r10));
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = _mm512_set1_pd(gate.i11);
                    hi_real_new = _mm512_fnmadd_pd(hi_imag, i11, _mm512_fmadd_pd(hi_real, r11, hi_real_new));
                    __m512d hi_imag_new = _mm512_fmadd_pd(lo_imag, r10, _mm512_mul_pd(lo_real, i10));
                    hi_imag_new = _mm512_fmadd_pd(hi_imag, r11, _mm512_fmadd_pd(hi_real, i11, hi_imag_new));
                    _mm512_i32scatter_pd(local_real, hi, hi_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, hi, hi_imag_new, 8);
                    idx = _mm256_add_epi32(idx, inc);
                }
                #else
                // TODO: switch (gate)
                int mask_inner = (1 << (LOCAL_QUBIT_SIZE - 1)) - (1 << gate.targetQubit);
                #pragma ivdep
                for (int j = 0; j < m; j++) {
                    int lo = j + (j & mask_inner);
                    int hi = lo | (1 << targetQubit);
                    cpx lo_val = cpx(local_real[lo], local_imag[lo]);
                    cpx hi_val = cpx(local_real[hi], local_imag[hi]);
                    cpx lo_val_new = lo_val * cpx(gate.r00, gate.i00) + hi_val * cpx(gate.r01, gate.i01);
                    local_real[lo] = lo_val_new.real();
                    local_imag[lo] = lo_val_new.imag();
                    cpx hi_val_new = lo_val * cpx(gate.r10, gate.i10) + hi_val * cpx(gate.r11, gate.i11);
                    local_real[hi] = hi_val_new.real();
                    local_imag[hi] = hi_val_new.imag();
                }
                #endif
            } else {
                bool isHighBlock = (blockID >> targetQubit) & 1;
                // TODO: switch (gate)
                int m = 1 << LOCAL_QUBIT_SIZE;
                #ifdef USE_AVX512
                __m512d rr = _mm512_set1_pd(isHighBlock ? gate.r11 : gate.r00);
                __m512d ii = _mm512_set1_pd(isHighBlock ? gate.i11 : gate.i00);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                for (int j = 0; j < m; j += 8) {
                    __m512d x_real = _mm512_i32gather_pd(idx, local_real, 8);
                    __m512d x_imag = _mm512_i32gather_pd(idx, local_imag, 8);
                    __m512d x_real_new = _mm512_fnmadd_pd(x_imag, ii, _mm512_mul_pd(x_real, rr));
                    __m512d x_imag_new = _mm512_fmadd_pd(x_imag, rr, _mm512_mul_pd(x_real, ii));
                    _mm512_i32scatter_pd(local_real, idx, x_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, idx, x_imag_new, 8);
                    idx = _mm256_add_epi32(idx, inc);
                }
                #else
                cpx param = isHighBlock ? cpx(gate.r11, gate.i11) : cpx(gate.r00, gate.i00);
                #pragma ivdep
                for (int j = 0; j < m; j++) {
                    cpx new_val = cpx(local_real[j], local_imag[j]) * param;
                    local_real[j] = new_val.real();
                    local_imag[j] = new_val.imag();
                }
                #endif
            }
        }
    }
}

void CpuExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    cpx* sv = deviceStateVec[0];
    #pragma omp parallel for
    for (int blockID = 0; blockID < (1 << (numLocalQubits - LOCAL_QUBIT_SIZE)); blockID++) {
        value_t local_real[1 << LOCAL_QUBIT_SIZE];
        value_t local_imag[1 << LOCAL_QUBIT_SIZE];
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
        fetch_data(local_real, local_imag, sv, bias, relatedQubits);
        apply_gate_group(local_real, local_imag, gates.size(), blockID, hostGates);
        save_data(sv, local_real, local_imag, bias, relatedQubits);
    }
    // for (int i = 0; i < gates.size(); i++) {
    //     printf("Gate %d [%d %d %lld] %d %d\n", hostGates[i].type, hostGates[i].encodeQubit, hostGates[i].controlQubit, hostGates[i].targetQubit, hostGates[i].controlIsGlobal, hostGates[i].targetIsGlobal);
    // }
    // printf("apply gate group: %f %f %f %f\n", sv[0].real(), sv[0].imag(), sv[1].real(), sv[1].imag());
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
            case GateType::RZZ: {
                int c = gate.encodeQubit;
                int t = gate.targetQubit;
                idx_t low_bit = std::min(c, t);
                idx_t high_bit = std::max(c, t);
                idx_t mask_inner = (idx_t(1) << low_bit) - 1;
                idx_t mask_middle = (idx_t(1) << (high_bit - 1)) - 1 - mask_inner;
                idx_t mask_outer = (idx_t(1) << (numLocalQubits - 2)) - 1 - mask_inner - mask_middle;
                #pragma omp for
                for (idx_t i = 0; i < (idx_t(1) << (numLocalQubits - 2)); i++) {
                    idx_t s00 = (i & mask_inner) + ((i & mask_middle) << 1) + ((i & mask_outer) << 2);
                    idx_t s01 = s00 | (idx_t(1) << t);
                    idx_t s10 = s00 | (idx_t(1) << c);
                    idx_t s11 = s01 | s10;
                    cpx v00 = deviceStateVec[0][s00];
                    cpx v11 = deviceStateVec[0][s11];
                    deviceStateVec[0][s00] = v00 * cpx(gate.r00, gate.i00) + v11 * cpx(gate.r11, gate.i11);
                    deviceStateVec[0][s11] = v11 * cpx(gate.r00, gate.i00) + v00 * cpx(gate.r11, gate.i11);

                    cpx v01 = deviceStateVec[0][s01];
                    cpx v10 = deviceStateVec[0][s10];
                    deviceStateVec[0][s01] = v01 * cpx(gate.r01, gate.i01) + v10 * cpx(gate.r10, gate.i10);
                    deviceStateVec[0][s10] = v10 * cpx(gate.r01, gate.i01) + v01 * cpx(gate.r10, gate.i10);
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

void CpuExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) {
    int numLocalQubits = numQubits - MyGlobalVars::bit;
    idx_t oldGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        oldGlobals |= 1ll << state.layout[i];
    idx_t newGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        newGlobals |= 1ll << newState.layout[i];
    
    idx_t globalMask = 0;
    idx_t localMasks[commSize];
    idx_t localMask = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        if (newState.layout[i] != state.layout[i]) {
            int x = state.layout[i];
            globalMask |= 1ll << i;
            localMask |= 1ll << newState.pos[x];
        }

    for (idx_t i = commSize-1, msk = localMask; i >= 0; i--, msk = localMask & (msk - 1)) {
        localMasks[i] = msk;
    }

    int sliceSize = 0;
    while (sliceSize < MAX_SLICE && !(localMask >> sliceSize & 1))
        sliceSize ++;

    cpx* tmpBuffer[MyGlobalVars::localGPUs];
    size_t tmpStart = 1ll << numLocalQubits;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++)
        tmpBuffer[i] = deviceStateVec[i] + tmpStart;

    for (idx_t iter = 0; iter < (1ll << numLocalQubits); iter += (1 << sliceSize)) {
        if (iter & localMask) continue;
        for (int xr = 1; xr < commSize; xr++) {
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                // the (a%commSize)-th GPU in the a/commSize comm_world (comm[a]) ->
                // the (a%commSize)^xr -th GPU in the same comm_world comm[a^xr]
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t srcBias = iter + localMasks[b % commSize];
#if USE_MPI
                int comm_a = comm[a] %  MyGlobalVars::localGPUs;
                checkMPIErrors(MPI_Sendrecv(
                    deviceStateVec[comm_a] + srcBias, 1 << sliceSize, MPI_Complex, comm[b], 0,
                    tmpBuffer[comm_a], 1 << sliceSize, MPI_Complex, comm[b], MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                ))
#else
                UNIMPLEMENTED();
#endif
            }
            // copy from tmp_buffer to dst
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[b] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t dstBias = iter + localMasks[a % commSize];
                int comm_b = comm[b] % MyGlobalVars::localGPUs;
                memcpy(deviceStateVec[comm_b] + dstBias, tmpBuffer[comm_b], (sizeof(cpx) << sliceSize));
            }
        }
    }
}

void CpuExecutor::dm_transpose()  { UNIMPLEMENTED(); }
void CpuExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CpuExecutor::eventBarrierAll() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

}
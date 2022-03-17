#include "cpu/cpu_dm_executor.h"
#include <assert.h>
#include <cstring>
#include <x86intrin.h>

namespace CpuImpl {

CpuDMExecutor::CpuDMExecutor(std::vector<cpx*> deviceStateVec, int numQubits, Schedule& schedule): DMExecutor(deviceStateVec, numQubits, schedule) {}

inline void fetch_data(value_t* local_real, value_t* local_imag, const cpx* deviceStateVec, int bias, idx_t related2) {
    int x;
    unsigned int y;
    idx_t mask = (1 << (COALESCE_GLOBAL * 2)) - 1;
    assert((related2 & mask) == mask);
    related2 -= mask;
    for (x = (1 << (LOCAL_QUBIT_SIZE * 2)) - 1 - mask, y = related2; x >= 0; x -= (1 << (COALESCE_GLOBAL * 2)), y = related2 & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << (COALESCE_GLOBAL * 2)); i++) {
            local_real[x + i] = deviceStateVec[(bias | y) + i].real();
            local_imag[x + i] = deviceStateVec[(bias | y) + i].imag();
            // printf("fetch %d <- %d\n", x + i, (bias | y) + i);
        }
    }
}

inline void save_data(cpx* deviceStateVec, const value_t* local_real, const value_t* local_imag, int bias, idx_t related2) {
    int x;
    unsigned int y;
    idx_t mask = (1 << (COALESCE_GLOBAL * 2)) - 1;
    assert((related2 & mask) == mask);
    related2 -= mask;
    for (x = (1 << (LOCAL_QUBIT_SIZE * 2)) - 1 - mask, y = related2; x >= 0; x -= (1 << COALESCE_GLOBAL * 2), y = related2 & (y-1)) {
        #pragma ivdep
        for (int i = 0; i < (1 << (COALESCE_GLOBAL * 2)); i++) {
            deviceStateVec[(bias | y) + i].real(local_real[x + i]);
            deviceStateVec[(bias | y) + i].imag(local_imag[x + i]);
        }
    }
}

#define CPXL(idx) (cpx(local_real[idx], local_imag[idx]))
#define CPXS(idx, val) {local_real[idx] = val.real(); local_imag[idx] = val.imag(); }
#define GATHER8(tr, ti, idx) __m512d tr = _mm512_i32gather_pd(idx, local_real, 8); __m512d ti = _mm512_i32gather_pd(idx, local_imag, 8);
#define SCATTER8(tr, ti, idx) _mm512_i32scatter_pd(local_real, idx, tr, 8); _mm512_i32scatter_pd(local_imag, idx, ti, 8);
#define NEW_ZERO_REG(reg) __m512d reg = _mm512_set1_pd(0.0);
#define LD_REG(tr, ti, val) __m512d tr = _mm512_set1_pd((val).real()); __m512d ti = _mm512_set1_pd((val).imag());

// t = a*b + c*d
// tr = ar * br - ai * bi + cr * dr - ci * di
// ti = ar * bi + ai * br + cr * di + ci * dr
#define CALC_AB_ADD_CD(tr, ti, ar, ai, br, bi, cr, ci, dr, di) \
    __m512d tr = _mm512_fnmadd_pd(ci, di, _mm512_fmadd_pd(cr, dr, _mm512_fnmadd_pd(ai, bi, _mm512_mul_pd(ar, br)))); \
    __m512d ti = _mm512_fmadd_pd(ci, dr, _mm512_fmadd_pd(cr, di, _mm512_fmadd_pd(ai, br, _mm512_mul_pd(ar, bi))));

// s = s + a*conj(b) + c*conj(d)
// sr = sr + ar * br + ai * bi + cr * dr + ci * di
// si = si - ar * bi + ai * br - cr * di + ci * dr
#define CALC_ADD_AB_ADD_CD_NT(sr, si, ar, ai, br, bi, cr, ci, dr, di) \
    sr = _mm512_fmadd_pd(ci, di, _mm512_fmadd_pd(cr, dr, _mm512_fmadd_pd(ai, bi, _mm512_fmadd_pd(ar, br, sr)))); \
    si = _mm512_fmadd_pd(ci, dr, _mm512_fnmadd_pd(cr, di, _mm512_fmadd_pd(ai, br, _mm512_fnmadd_pd(ar, bi, si))));

#ifdef USE_AVX512

void dbgv(const char* name, __m256i reg) {
    int val[8];
    memcpy(val, &reg, sizeof(reg));
    printf("%s: %d %d %d %d %d %d %d %d\n", name, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}
void dbgv(const char* name, __m512d reg) {
    double val[8];
    memcpy(val, &reg, sizeof(reg));
    printf("%s: %f %f %f %f %f %f %f %f\n", name, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}
#endif

inline void apply_gate_group(value_t* local_real, value_t* local_imag, int numGates, int blockID, KernelGate hostGates[]) {
#if MODE == 2
    constexpr int local2 = LOCAL_QUBIT_SIZE * 2;
    for (int i = 0; i < numGates; i++) {
        auto& gate = hostGates[i];
        int controlQubit = gate.controlQubit;
        int targetQubit = gate.targetQubit;

        if (gate.controlQubit == -1) { // single qubit gate
            // skip due to error fusion
        } else {
            if (gate.controlQubit == -3) { // two qubit gate
                controlQubit = gate.encodeQubit;
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (local2 - 2);
                int low_bit = std::min(controlQubit, targetQubit);
                int high_bit = std::max(controlQubit, targetQubit);
#ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << low_bit));
                __m256i mask_outer = _mm256_set1_epi32((1 << (local2 - 1)) - (1 << high_bit));
                __m256i ctr_flag = _mm256_set1_epi32(1 << controlQubit);
                __m256i tar_flag = _mm256_set1_epi32(1 << targetQubit);
                __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                const __m256i inc = _mm256_set1_epi32(8);
                for (int j = 0; j < m; j += 8) {
                    __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                    lo = _mm256_add_epi32(lo, _mm256_and_si256(lo, mask_outer));
                    __m256i s00 = lo;
                    __m256i s01 = _mm256_add_epi32(s00, ctr_flag);
                    __m256i s10 = _mm256_add_epi32(s00, tar_flag);
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
                low_bit++; high_bit++;
                mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << low_bit));
                mask_outer = _mm256_set1_epi32((1 << (local2 - 1)) - (1 << high_bit));
                ctr_flag = _mm256_set1_epi32(1 << (controlQubit + 1));
                tar_flag = _mm256_set1_epi32(1 << (targetQubit + 1));
                idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                for (int j = 0; j < m; j += 8) {
                    __m256i lo = _mm256_add_epi32(idx, _mm256_and_si256(idx, mask_inner));
                    lo = _mm256_add_epi32(lo, _mm256_and_si256(lo, mask_outer));
                    __m256i s00 = lo;
                    __m256i s01 = _mm256_add_epi32(s00, ctr_flag);
                    __m256i s10 = _mm256_add_epi32(s00, tar_flag);
                    __m256i s11 = _mm256_or_si256(s01, s10);

                    __m512d v00_real = _mm512_i32gather_pd(s00, local_real, 8);
                    __m512d v00_imag = _mm512_i32gather_pd(s00, local_imag, 8);
                    __m512d v11_real = _mm512_i32gather_pd(s11, local_real, 8);
                    __m512d v11_imag = _mm512_i32gather_pd(s11, local_imag, 8);
                    __m512d r00 = _mm512_set1_pd(gate.r00);
                    __m512d i00 = -_mm512_set1_pd(gate.i00);
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = -_mm512_set1_pd(gate.i11);
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
                    __m512d i01 = -_mm512_set1_pd(gate.i01);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = -_mm512_set1_pd(gate.i10);
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
                int mask_inner = (1 << (local2 - 2)) - (1 << low_bit);
                int mask_outer = (1 << (local2 - 1)) - (1 << high_bit);
                for (int j = 0; j < m; j++) {
                    int s00 = j + (j & mask_inner);
                    s00 = s00 + (s00 & mask_outer);
                    int s01 = s00 | (1 << controlQubit);
                    int s10 = s00 | (1 << targetQubit);
                    int s11 = s01 | s10;

                    cpx val00 = CPXL(s00);
                    cpx val01 = CPXL(s01);
                    cpx val10 = CPXL(s10);
                    cpx val11 = CPXL(s11);

                    cpx val00_new = val00 * cpx(gate.r00, gate.i00) + val11 * cpx(gate.r11, gate.i11);
                    cpx val01_new = val01 * cpx(gate.r01, gate.i01) + val10 * cpx(gate.r10, gate.i10);
                    cpx val10_new = val01 * cpx(gate.r10, gate.i10) + val10 * cpx(gate.r01, gate.i01);
                    cpx val11_new = val00 * cpx(gate.r11, gate.i11) + val11 * cpx(gate.r00, gate.i00);

                    CPXS(s00, val00_new)
                    CPXS(s01, val01_new)
                    CPXS(s10, val10_new)
                    CPXS(s11, val11_new)
                }
                low_bit++; high_bit++;
                mask_inner = (1 << (local2 - 2)) - (1 << low_bit);
                mask_outer = (1 << (local2 - 1)) - (1 << high_bit);

                for (int j = 0; j < m; j++) {
                    int s00 = j + (j & mask_inner);
                    s00 = s00 + (s00 & mask_outer);
                    int s01 = s00 | (1 << (controlQubit + 1));
                    int s10 = s00 | (1 << (targetQubit + 1));
                    int s11 = s01 | s10;

                    cpx val00 = CPXL(s00);
                    cpx val01 = CPXL(s01);
                    cpx val10 = CPXL(s10);
                    cpx val11 = CPXL(s11);

                    cpx val00_new = val00 * cpx(gate.r00, -gate.i00) + val11 * cpx(gate.r11, -gate.i11);
                    cpx val01_new = val01 * cpx(gate.r01, -gate.i01) + val10 * cpx(gate.r10, -gate.i10);
                    cpx val10_new = val01 * cpx(gate.r10, -gate.i10) + val10 * cpx(gate.r01, -gate.i01);
                    cpx val11_new = val00 * cpx(gate.r11, -gate.i11) + val11 * cpx(gate.r00, -gate.i00);

                    CPXS(s00, val00_new)
                    CPXS(s01, val01_new)
                    CPXS(s10, val10_new)
                    CPXS(s11, val11_new)
                }
#endif
            } else { // controlled gate
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (local2 - 2);
                int low_bit = std::min(controlQubit, targetQubit);
                int high_bit = std::max(controlQubit, targetQubit);
#ifdef USE_AVX512
                __m256i mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << low_bit));
                __m256i mask_outer = _mm256_set1_epi32((1 << (local2 - 1)) - (1 << high_bit));
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
                low_bit++; high_bit++;
                mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << low_bit));
                mask_outer = _mm256_set1_epi32((1 << (local2 - 1)) - (1 << high_bit));
                ctr_flag = _mm256_set1_epi32(1 << (controlQubit + 1));
                tar_flag = _mm256_set1_epi32(1 << (targetQubit + 1));
                assert(m % 8 == 0);
                idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
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
                    __m512d i00 = -_mm512_set1_pd(gate.i00);
                    __m512d r01 = _mm512_set1_pd(gate.r01);
                    __m512d i01 = -_mm512_set1_pd(gate.i01);
                    __m512d lo_real_new = _mm512_fnmadd_pd(lo_imag, i00, _mm512_mul_pd(lo_real, r00));
                    lo_real_new = _mm512_fnmadd_pd(hi_imag, i01, _mm512_fmadd_pd(hi_real, r01, lo_real_new));
                    __m512d lo_imag_new = _mm512_fmadd_pd(lo_imag, r00, _mm512_mul_pd(lo_real, i00));
                    lo_imag_new = _mm512_fmadd_pd(hi_imag, r01, _mm512_fmadd_pd(hi_real, i01, lo_imag_new));
                    _mm512_i32scatter_pd(local_real, lo, lo_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, lo, lo_imag_new, 8);
                    __m512d r10 = _mm512_set1_pd(gate.r10);
                    __m512d i10 = -_mm512_set1_pd(gate.i10);
                    __m512d hi_real_new = _mm512_fnmadd_pd(lo_imag, i10, _mm512_mul_pd(lo_real, r10));
                    __m512d r11 = _mm512_set1_pd(gate.r11);
                    __m512d i11 = -_mm512_set1_pd(gate.i11);
                    hi_real_new = _mm512_fnmadd_pd(hi_imag, i11, _mm512_fmadd_pd(hi_real, r11, hi_real_new));
                    __m512d hi_imag_new = _mm512_fmadd_pd(lo_imag, r10, _mm512_mul_pd(lo_real, i10));
                    hi_imag_new = _mm512_fmadd_pd(hi_imag, r11, _mm512_fmadd_pd(hi_real, i11, hi_imag_new));
                    _mm512_i32scatter_pd(local_real, hi, hi_real_new, 8);
                    _mm512_i32scatter_pd(local_imag, hi, hi_imag_new, 8);
                    idx = _mm256_add_epi32(idx, inc);
                }
#else
                int mask_inner = (1 << (local2 - 2)) - (1 << low_bit);
                int mask_outer = (1 << (local2 - 1)) - (1 << high_bit);
                for (int j = 0; j < m; j++) {
                    int s0 = j + (j & mask_inner);
                    s0 = s0 + (s0 & mask_outer);
                    s0 |= (1 << controlQubit);
                    int s1 = s0 | (1 << targetQubit);
                    cpx val0 = CPXL(s0);
                    cpx val1 = CPXL(s1);
                    cpx val0_new = val0 * cpx(gate.r00, gate.i00) + val1 * cpx(gate.r01, gate.i01);
                    cpx val1_new = val0 * cpx(gate.r10, gate.i10) + val1 * cpx(gate.r11, gate.i11);
                    CPXS(s0, val0_new)
                    CPXS(s1, val1_new)
                }

                low_bit++; high_bit++;
                mask_inner = (1 << (local2 - 2)) - (1 << low_bit);
                mask_outer = (1 << (local2 - 1)) - (1 << high_bit);

                for (int j = 0; j < m; j++) {
                    int s0 = j + (j & mask_inner);
                    s0 = s0 + (s0 & mask_outer);
                    s0 |= (1 << (controlQubit + 1));
                    int s1 = s0 | (1 << (targetQubit + 1));
                    cpx val0 = CPXL(s0);
                    cpx val1 = CPXL(s1);
                    cpx val0_new = val0 * cpx(gate.r00, -gate.i00) + val1 * cpx(gate.r01, -gate.i01);
                    cpx val1_new = val0 * cpx(gate.r10, -gate.i10) + val1 * cpx(gate.r11, -gate.i11);
                    CPXS(s0, val0_new)
                    CPXS(s1, val1_new)
                }
#endif
            }
        }
        if (hostGates[i].err_len_target > 0) {
            int m = 1 << (local2 - 2);
            int qid = hostGates[i].targetQubit * 2;
            int numErrors = hostGates[i].err_len_target;
#ifdef USE_AVX512
            __m256i mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << qid));
            const __m256i inc = _mm256_set1_epi32(8);
            const __m256i mul = _mm256_set1_epi32(3);
            const __m256i inner_flag = _mm256_set1_epi32(1 << qid);
            const __m256i outer_flag = _mm256_set1_epi32(1 << (qid + 1));
            __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            for (int j = 0; j < m; j += 8) {
                __m256i tmp = _mm256_and_si256(idx, mask_inner);
                __m256i s00 = _mm256_add_epi32(idx, _mm256_add_epi32(tmp, _mm256_add_epi32(tmp, tmp))); // _mm256_mul_epi32 only multiplies 4 values
                __m256i s01 = _mm256_add_epi32(s00, inner_flag);
                __m256i s10 = _mm256_add_epi32(s00, outer_flag);
                __m256i s11 = _mm256_or_si256(s01, s10);
                GATHER8(val00_real, val00_imag, s00)
                GATHER8(val01_real, val01_imag, s01)
                GATHER8(val10_real, val10_imag, s10)
                GATHER8(val11_real, val11_imag, s11)
                NEW_ZERO_REG(sum00_real)
                NEW_ZERO_REG(sum00_imag)
                NEW_ZERO_REG(sum01_real)
                NEW_ZERO_REG(sum01_imag)
                NEW_ZERO_REG(sum10_real)
                NEW_ZERO_REG(sum10_imag)
                NEW_ZERO_REG(sum11_real)
                NEW_ZERO_REG(sum11_imag)

                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_target[k];
                    LD_REG(e00_real, e00_imag, e[0][0]);
                    LD_REG(e01_real, e01_imag, e[0][1]);
                    LD_REG(e10_real, e10_imag, e[1][0]);
                    LD_REG(e11_real, e11_imag, e[1][1]);
                    CALC_AB_ADD_CD(w00_real, w00_imag, e00_real, e00_imag, val00_real, val00_imag, e01_real, e01_imag, val10_real, val10_imag);
                    CALC_AB_ADD_CD(w01_real, w01_imag, e00_real, e00_imag, val01_real, val01_imag, e01_real, e01_imag, val11_real, val11_imag);
                    CALC_AB_ADD_CD(w10_real, w10_imag, e10_real, e10_imag, val00_real, val00_imag, e11_real, e11_imag, val10_real, val10_imag);
                    CALC_AB_ADD_CD(w11_real, w11_imag, e10_real, e10_imag, val01_real, val01_imag, e11_real, e11_imag, val11_real, val11_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum00_real, sum00_imag, w00_real, w00_imag, e00_real, e00_imag, w01_real, w01_imag, e01_real, e01_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum01_real, sum01_imag, w00_real, w00_imag, e10_real, e10_imag, w01_real, w01_imag, e11_real, e11_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum10_real, sum10_imag, w10_real, w10_imag, e00_real, e00_imag, w11_real, w11_imag, e01_real, e01_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum11_real, sum11_imag, w10_real, w10_imag, e10_real, e10_imag, w11_real, w11_imag, e11_real, e11_imag);
                }
                SCATTER8(sum00_real, sum00_imag, s00);
                SCATTER8(sum01_real, sum01_imag, s01);
                SCATTER8(sum10_real, sum10_imag, s10);
                SCATTER8(sum11_real, sum11_imag, s11);
                idx = _mm256_add_epi32(idx, inc);
            }
#else
            int mask_inner = (1 << (local2 - 2)) - (1 << qid);
            for (int j = 0; j < m; j++) {
                int s00 = j + (j & mask_inner) * 3;
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                cpx val00 = CPXL(s00);
                cpx val01 = CPXL(s01);
                cpx val10 = CPXL(s10);
                cpx val11 = CPXL(s11);

                cpx sum00 = cpx(0.0), sum01 = cpx(0.0), sum10 = cpx(0.0), sum11=cpx(0.0);
                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_target[k];
                    cpx w00 = e[0][0] * val00 + e[0][1] * val10;
                    cpx w01 = e[0][0] * val01 + e[0][1] * val11;
                    cpx w10 = e[1][0] * val00 + e[1][1] * val10;
                    cpx w11 = e[1][0] * val01 + e[1][1] * val11;
                    sum00 += w00 * std::conj(e[0][0]) + w01 * std::conj(e[0][1]);
                    sum01 += w00 * std::conj(e[1][0]) + w01 * std::conj(e[1][1]);
                    sum10 += w10 * std::conj(e[0][0]) + w11 * std::conj(e[0][1]);
                    sum11 += w10 * std::conj(e[1][0]) + w11 * std::conj(e[1][1]);
                }

                CPXS(s00, sum00)
                CPXS(s01, sum01)
                CPXS(s10, sum10)
                CPXS(s11, sum11)
            }
#endif
        }
        if (hostGates[i].err_len_control > 0) {
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
            int qid = hostGates[i].controlQubit == -3? hostGates[i].encodeQubit: hostGates[i].controlQubit;
            qid *= 2;
            int numErrors = hostGates[i].err_len_control;
#ifdef USE_AVX512
            __m256i mask_inner = _mm256_set1_epi32((1 << (local2 - 2)) - (1 << qid));
            const __m256i inc = _mm256_set1_epi32(8);
            const __m256i mul = _mm256_set1_epi32(3);
            const __m256i inner_flag = _mm256_set1_epi32(1 << qid);
            const __m256i outer_flag = _mm256_set1_epi32(1 << (qid + 1));
            __m256i idx = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            for (int j = 0; j < m; j += 8) {
                __m256i tmp = _mm256_and_si256(idx, mask_inner);
                __m256i s00 = _mm256_add_epi32(idx, _mm256_add_epi32(tmp, _mm256_add_epi32(tmp, tmp))); // _mm256_mul_epi32 only multiplies 4 values
                __m256i s01 = _mm256_add_epi32(s00, inner_flag);
                __m256i s10 = _mm256_add_epi32(s00, outer_flag);
                __m256i s11 = _mm256_or_si256(s01, s10);
                GATHER8(val00_real, val00_imag, s00)
                GATHER8(val01_real, val01_imag, s01)
                GATHER8(val10_real, val10_imag, s10)
                GATHER8(val11_real, val11_imag, s11)
                NEW_ZERO_REG(sum00_real)
                NEW_ZERO_REG(sum00_imag)
                NEW_ZERO_REG(sum01_real)
                NEW_ZERO_REG(sum01_imag)
                NEW_ZERO_REG(sum10_real)
                NEW_ZERO_REG(sum10_imag)
                NEW_ZERO_REG(sum11_real)
                NEW_ZERO_REG(sum11_imag)

                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_control[k];
                    LD_REG(e00_real, e00_imag, e[0][0]);
                    LD_REG(e01_real, e01_imag, e[0][1]);
                    LD_REG(e10_real, e10_imag, e[1][0]);
                    LD_REG(e11_real, e11_imag, e[1][1]);
                    CALC_AB_ADD_CD(w00_real, w00_imag, e00_real, e00_imag, val00_real, val00_imag, e01_real, e01_imag, val10_real, val10_imag);
                    CALC_AB_ADD_CD(w01_real, w01_imag, e00_real, e00_imag, val01_real, val01_imag, e01_real, e01_imag, val11_real, val11_imag);
                    CALC_AB_ADD_CD(w10_real, w10_imag, e10_real, e10_imag, val00_real, val00_imag, e11_real, e11_imag, val10_real, val10_imag);
                    CALC_AB_ADD_CD(w11_real, w11_imag, e10_real, e10_imag, val01_real, val01_imag, e11_real, e11_imag, val11_real, val11_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum00_real, sum00_imag, w00_real, w00_imag, e00_real, e00_imag, w01_real, w01_imag, e01_real, e01_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum01_real, sum01_imag, w00_real, w00_imag, e10_real, e10_imag, w01_real, w01_imag, e11_real, e11_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum10_real, sum10_imag, w10_real, w10_imag, e00_real, e00_imag, w11_real, w11_imag, e01_real, e01_imag);
                    CALC_ADD_AB_ADD_CD_NT(sum11_real, sum11_imag, w10_real, w10_imag, e10_real, e10_imag, w11_real, w11_imag, e11_real, e11_imag);
                }
                SCATTER8(sum00_real, sum00_imag, s00);
                SCATTER8(sum01_real, sum01_imag, s01);
                SCATTER8(sum10_real, sum10_imag, s10);
                SCATTER8(sum11_real, sum11_imag, s11);
                idx = _mm256_add_epi32(idx, inc);
            }
#else
            int mask_inner = (1 << (local2 - 2)) - (1 << qid);
            for (int j = 0; j < m; j++) {
                int s00 = j + (j & mask_inner) * 3;
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                cpx val00 = CPXL(s00);
                cpx val01 = CPXL(s01);
                cpx val10 = CPXL(s10);
                cpx val11 = CPXL(s11);

                cpx sum00 = cpx(0.0), sum01 = cpx(0.0), sum10 = cpx(0.0), sum11=cpx(0.0);
                for (int k = 0; k < numErrors; k++) {
                    cpx (*e)[2] = hostGates[i].errs_control[k];
                    cpx w00 = e[0][0] * val00 + e[0][1] * val10;
                    cpx w01 = e[0][0] * val01 + e[0][1] * val11;
                    cpx w10 = e[1][0] * val00 + e[1][1] * val10;
                    cpx w11 = e[1][0] * val01 + e[1][1] * val11;
                    sum00 += w00 * std::conj(e[0][0]) + w01 * std::conj(e[0][1]);
                    sum01 += w00 * std::conj(e[1][0]) + w01 * std::conj(e[1][1]);
                    sum10 += w10 * std::conj(e[0][0]) + w11 * std::conj(e[0][1]);
                    sum11 += w10 * std::conj(e[1][0]) + w11 * std::conj(e[1][1]);
                }

                CPXS(s00, sum00)
                CPXS(s01, sum01)
                CPXS(s10, sum10)
                CPXS(s11, sum11)
            }
#endif
        }
    }
#else
    UNREACHABLE();
#endif
}

void CpuDMExecutor::launchPerGateGroupDM(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) {
    cpx* sv = deviceStateVec[0];
    unsigned int related2 = duplicate_bit(relatedQubits); // warning: put it in master core
    idx_t blockHot = (idx_t(1) << (numLocalQubits * 2)) - 1 - related2;
    #pragma omp parallel for
    for (int blockID = 0; blockID < (1 << ((numLocalQubits - LOCAL_QUBIT_SIZE) * 2)); blockID++) {
        value_t local_real[1 << (LOCAL_QUBIT_SIZE * 2)];
        value_t local_imag[1 << (LOCAL_QUBIT_SIZE * 2)];
        unsigned int bias = 0;
        {
            int bid = blockID;
            for (unsigned int bit = 1; bit < (1u << (numLocalQubits * 2)); bit <<= 1) {
                if (blockHot & bit) {
                    if (bid & 1)
                        bias |= bit;
                    bid >>= 1;
                }
            }
        }
        fetch_data(local_real, local_imag, sv, bias, related2);
        apply_gate_group(local_real, local_imag, gates.size(), blockID, hostGates);
        save_data(sv, local_real, local_imag, bias, related2);
    }
}

void CpuDMExecutor::dm_transpose() { UNIMPLEMENTED(); }

void CpuDMExecutor::transpose(std::vector<std::shared_ptr<hptt::Transpose<cpx>>> plans) {
    plans[0]->setInputPtr(deviceStateVec[0]);
    plans[0]->setOutputPtr(deviceBuffer[0]);
    plans[0]->execute();
}

void CpuDMExecutor::inplaceAll2All(int commSize, std::vector<int> comm, const State& newState) {
    int numLocalQubits = numQubits - MyGlobalVars::bit / 2;
    idx_t oldGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        oldGlobals |= 1ll << state.layout[i];
    idx_t newGlobals = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        newGlobals |= 1ll << newState.layout[i];
    
    idx_t localMasks[commSize];
    idx_t localMask = 0;
    for (int i = numLocalQubits; i < numQubits; i++)
        if (newState.layout[i] != state.layout[i]) {
            int x = state.layout[i];
            localMask |= 1ll << newState.pos[x];
        }
    

    idx_t sliceSize = 0;
    while (sliceSize < MAX_SLICE && !(localMask >> sliceSize & 1))
        sliceSize ++;
    sliceSize = idx_t(1) << (sliceSize * 2);

    localMask = duplicate_bit(localMask);
    for (idx_t i = commSize-1, msk = localMask; i >= 0; i--, msk = localMask & (msk - 1)) {
        localMasks[i] = msk;
    }

    cpx* tmpBuffer[MyGlobalVars::localGPUs];
    size_t tmpStart = 1ll << (numLocalQubits * 2);
    if (GPU_BACKEND == 3 || GPU_BACKEND == 4)
        tmpStart <<= 1;
    for (int i = 0; i < MyGlobalVars::localGPUs; i++)
        tmpBuffer[i] = deviceStateVec[i] + tmpStart;

    for (idx_t iter = 0; iter < (1ll << (numLocalQubits * 2)); iter += sliceSize) {
        if (iter & localMask) continue;
        for (int xr = 1; xr < commSize; xr++) {
            // copy from src to tmp_buffer
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                // the (a%commSize)-th GPU in the a/commSize comm_world (comm[a]) ->
                // the (a%commSize)^xr-th GPU in the same comm_world comm[a^xr]
                int b = a ^ xr;
                if (comm[a] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t srcBias = iter + localMasks[b % commSize];
#if USE_MPI
                int comm_a = comm[a] %  MyGlobalVars::localGPUs;
                checkMPIErrors(MPI_Sendrecv(
                    deviceStateVec[comm_a] + srcBias, sliceSize, MPI_Complex, comm[b], 0,
                    tmpBuffer[comm_a], sliceSize, MPI_Complex, comm[b], MPI_ANY_TAG,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                ));
#else
                UNREACHABLE();
#endif
            }
            // copy from tmp_buffer to dst
            for (int a = 0; a < MyGlobalVars::numGPUs; a++) {
                int b = a ^ xr;
                if (comm[b] / MyGlobalVars::localGPUs != MyMPI::rank)
                    continue;
                idx_t dstBias = iter + localMasks[a % commSize];
                int comm_b = comm[b] % MyGlobalVars::localGPUs;
                memcpy(deviceStateVec[comm_b] + dstBias, tmpBuffer[comm_b], (sizeof(cpx) * sliceSize));
            }
        }
    }
}

void CpuDMExecutor::all2all(int commSize, std::vector<int> comm) {
    int numLocalQubit = numQubits - MyGlobalVars::bit / 2;
    idx_t numElements = 1ll << (numLocalQubit * 2);
    int numPart = numSlice / commSize;
#ifdef ALL_TO_ALL
    #if USE_MPI
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
    ));
    #else
    UNIMPLEMENTED();
    #endif
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
            }
        }
    }
#endif
}

void CpuDMExecutor::launchPerGateGroup(std::vector<Gate>& gates, KernelGate hostGates[], const State& state, idx_t relatedQubits, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchPerGateGroupSliced(std::vector<Gate>& gates, KernelGate hostGates[], idx_t relatedQubits, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchBlasGroup(GateGroup& gg, int numLocalQubits) { UNIMPLEMENTED(); }
void CpuDMExecutor::launchBlasGroupSliced(GateGroup& gg, int numLocalQubits, int sliceID) { UNIMPLEMENTED(); }

void CpuDMExecutor::deviceFinalize() {}

void CpuDMExecutor::sliceBarrier(int sliceID) { UNIMPLEMENTED(); }
void CpuDMExecutor::eventBarrier() { UNIMPLEMENTED(); }
void CpuDMExecutor::eventBarrierAll() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

void CpuDMExecutor::allBarrier() {
#if USE_MPI
    checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
#endif
}

}
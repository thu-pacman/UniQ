#include "cuda/kernel.h"
#include <cstdio>
#include <assert.h>
#include <map>
#include <omp.h>
#include "gate.h"
#include "executor.h"
#include "dbg.h"
using namespace std;

namespace CudaDM {
std::vector<KernelGate*> deviceGatePointers;

#define COMPLEX_MULTIPLY_REAL(v0, v1) (v0.x * v1.x - v0.y * v1.y)
#define COMPLEX_MULTIPLY_IMAG(v0, v1) (v0.x * v1.y + v0.y * v1.x)

// ND: v1.y -> -v1.y
#define COMPLEX_MULTIPLY_REAL_ND(v0, v1) (v0.x * v1.x + v0.y * v1.y)
#define COMPLEX_MULTIPLY_IMAG_ND(v0, v1) (-v0.x * v1.y + v0.y * v1.x)

#if MODE==2
static __shared__ cuCpx shm[1<<(LOCAL_QUBIT_SIZE * 2)];
static __shared__ idx_t blockBias;

__device__ __constant__ value_t recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
template <unsigned int blockSize>

__device__ void doComputeDM(int numGates, KernelGate* deviceGates) {
    for (int i = 0; i < numGates; i++) {
        int controlQubit = deviceGates[i].controlQubit;
        int targetQubit = deviceGates[i].targetQubit;
        if (deviceGates[i].controlQubit == -1) { // single qubit gate
            // skip due to gate fusion
        } else {
            if (deviceGates[i].controlQubit == -3) { // twoQubitGate
                auto& gate = deviceGates[i];
                controlQubit = gate.encodeQubit;
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
                // U\rho
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
                    int s00 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s00 = ((s00 >> largeQubit) << (largeQubit + 1)) | (s00 & maskLarge);
                    int s01 = s00 | (1 << controlQubit);
                    int s10 = s00 | (1 << targetQubit);
                    int s11 = s01 | s10;

                    cuCpx val_00 = shm[s00];
                    cuCpx val_01 = shm[s01];
                    cuCpx val_10 = shm[s10];
                    cuCpx val_11 = shm[s11];
                    shm[s00] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_00, make_cuComplex(gate.r00, gate.i00)) + COMPLEX_MULTIPLY_REAL(val_11, make_cuComplex(gate.r11, gate.i11)),
                        COMPLEX_MULTIPLY_IMAG(val_00, make_cuComplex(gate.r00, gate.i00)) + COMPLEX_MULTIPLY_IMAG(val_11, make_cuComplex(gate.r11, gate.i11))
                    );
                    shm[s01] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_01, make_cuComplex(gate.r01, gate.i01)) + COMPLEX_MULTIPLY_REAL(val_10, make_cuComplex(gate.r10, gate.i10)),
                        COMPLEX_MULTIPLY_IMAG(val_01, make_cuComplex(gate.r01, gate.i01)) + COMPLEX_MULTIPLY_IMAG(val_10, make_cuComplex(gate.r10, gate.i10))
                    );
                    shm[s10] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_01, make_cuComplex(gate.r10, gate.i10)) + COMPLEX_MULTIPLY_REAL(val_10, make_cuComplex(gate.r01, gate.i01)),
                        COMPLEX_MULTIPLY_IMAG(val_01, make_cuComplex(gate.r10, gate.i10)) + COMPLEX_MULTIPLY_IMAG(val_10, make_cuComplex(gate.r01, gate.i01))
                    );
                    shm[s11] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_00, make_cuComplex(gate.r11, gate.i11)) + COMPLEX_MULTIPLY_REAL(val_11, make_cuComplex(gate.r00, gate.i00)),
                        COMPLEX_MULTIPLY_IMAG(val_00, make_cuComplex(gate.r11, gate.i11)) + COMPLEX_MULTIPLY_IMAG(val_11, make_cuComplex(gate.r00, gate.i00))
                    );
                }
                __syncthreads();
                // \rho U^\dagger
                smallQubit ++; largeQubit++;
                maskSmall = (1 << smallQubit) - 1;
                maskLarge = (1 << largeQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
                    int s00 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s00 = ((s00 >> largeQubit) << (largeQubit + 1)) | (s00 & maskLarge);
                    int s01 = s00 | (1 << (targetQubit + 1));
                    int s10 = s00 | (1 << (controlQubit + 1));
                    int s11 = s01 | s10;
                    cuCpx val_00 = shm[s00];
                    cuCpx val_01 = shm[s01];
                    cuCpx val_10 = shm[s10];
                    cuCpx val_11 = shm[s11];
                    shm[s00] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_00, make_cuComplex(gate.r00, -gate.i00)) + COMPLEX_MULTIPLY_REAL(val_11, make_cuComplex(gate.r11, -gate.i11)),
                        COMPLEX_MULTIPLY_IMAG(val_00, make_cuComplex(gate.r00, -gate.i00)) + COMPLEX_MULTIPLY_IMAG(val_11, make_cuComplex(gate.r11, -gate.i11))
                    );
                    shm[s01] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_01, make_cuComplex(gate.r01, -gate.i01)) + COMPLEX_MULTIPLY_REAL(val_10, make_cuComplex(gate.r10, -gate.i10)),
                        COMPLEX_MULTIPLY_IMAG(val_01, make_cuComplex(gate.r01, -gate.i01)) + COMPLEX_MULTIPLY_IMAG(val_10, make_cuComplex(gate.r10, -gate.i10))
                    );
                    shm[s10] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_01, make_cuComplex(gate.r10, -gate.i10)) + COMPLEX_MULTIPLY_REAL(val_10, make_cuComplex(gate.r01, -gate.i01)),
                        COMPLEX_MULTIPLY_IMAG(val_01, make_cuComplex(gate.r10, -gate.i10)) + COMPLEX_MULTIPLY_IMAG(val_10, make_cuComplex(gate.r01, -gate.i01))
                    );
                    shm[s11] =  make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val_00, make_cuComplex(gate.r11, -gate.i11)) + COMPLEX_MULTIPLY_REAL(val_11, make_cuComplex(gate.r00, -gate.i00)),
                        COMPLEX_MULTIPLY_IMAG(val_00, make_cuComplex(gate.r11, -gate.i11)) + COMPLEX_MULTIPLY_IMAG(val_11, make_cuComplex(gate.r00, -gate.i00))
                    );
                }
            } else { // controlled gate
                auto& gate = deviceGates[i];
                controlQubit *= 2;
                targetQubit *= 2;
                int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
                // U\rho
                int smallQubit = controlQubit > targetQubit ? targetQubit : controlQubit;
                int largeQubit = controlQubit > targetQubit ? controlQubit : targetQubit;
                int maskSmall = (1 << smallQubit) - 1;
                int maskLarge = (1 << largeQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
                    int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s0 = ((s0 >> largeQubit) << (largeQubit + 1)) | (s0 & maskLarge);
                    s0 |= (1 << controlQubit);
                    int s1 = s0 | (1 << targetQubit);
                    cuCpx val0 = shm[s0];
                    cuCpx val1 = shm[s1];
                    shm[s0] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val0, make_cuComplex(gate.r00, gate.i00)) + COMPLEX_MULTIPLY_REAL(val1, make_cuComplex(gate.r01, gate.i01)),
                        COMPLEX_MULTIPLY_IMAG(val0, make_cuComplex(gate.r00, gate.i00)) + COMPLEX_MULTIPLY_IMAG(val1, make_cuComplex(gate.r01, gate.i01))
                    );
                    shm[s1] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val0, make_cuComplex(gate.r10, gate.i10)) + COMPLEX_MULTIPLY_REAL(val1, make_cuComplex(gate.r11, gate.i11)),
                        COMPLEX_MULTIPLY_IMAG(val0, make_cuComplex(gate.r10, gate.i10)) + COMPLEX_MULTIPLY_IMAG(val1, make_cuComplex(gate.r11, gate.i11))
                    );
                }
                __syncthreads();
                // \rho U^\dagger
                smallQubit ++; largeQubit++;
                maskSmall = (1 << smallQubit) - 1;
                maskLarge = (1 << largeQubit) - 1;
                for (int j = threadIdx.x; j < m; j += blockSize) {
                    int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskSmall);
                    s0 = ((s0 >> largeQubit) << (largeQubit + 1)) | (s0 & maskLarge);
                    s0 |= 1 << (controlQubit + 1);
                    int s1 = s0 | (1 << (targetQubit + 1));
                    cuCpx val0 = shm[s0];
                    cuCpx val1 = shm[s1];
                    shm[s0] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val0, make_cuComplex(gate.r00, -gate.i00)) + COMPLEX_MULTIPLY_REAL(val1, make_cuComplex(gate.r01, -gate.i01)),
                        COMPLEX_MULTIPLY_IMAG(val0, make_cuComplex(gate.r00, -gate.i00)) + COMPLEX_MULTIPLY_IMAG(val1, make_cuComplex(gate.r01, -gate.i01))
                    );
                    shm[s1] = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(val0, make_cuComplex(gate.r10, -gate.i10)) + COMPLEX_MULTIPLY_REAL(val1, make_cuComplex(gate.r11, -gate.i11)),
                        COMPLEX_MULTIPLY_IMAG(val0, make_cuComplex(gate.r10, -gate.i10)) + COMPLEX_MULTIPLY_IMAG(val1, make_cuComplex(gate.r11, -gate.i11))
                    );
                }
            }
        }
        // apply error
        __syncthreads();
        if (deviceGates[i].err_len_target > 0) {
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
            int qid = deviceGates[i].targetQubit * 2;
            int numErrors = deviceGates[i].err_len_target;
            for (int j = threadIdx.x; j < m; j += blockSize) {
                int s00 = ((j >> qid) << (qid + 2)) | (j & ((1 << qid) - 1));
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                cuCpx val00 = shm[s00], val01 = shm[s01], val10 = shm[s10], val11 = shm[s11];
                cuCpx sum00 = make_cuComplex(0.0, 0.0), sum01 = make_cuComplex(0.0, 0.0), sum10 = make_cuComplex(0.0, 0.0), sum11 = make_cuComplex(0.0, 0.0);
                for (int k = 0; k < numErrors; k++) {
                    cuCpx (*e)[2] = reinterpret_cast<cuCpx(*)[2]>(deviceGates[i].errs_target[k]);
                    cuCpx w00 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[0][0], val00) + COMPLEX_MULTIPLY_REAL(e[0][1], val10),
                        COMPLEX_MULTIPLY_IMAG(e[0][0], val00) + COMPLEX_MULTIPLY_IMAG(e[0][1], val10)
                    ); // e.mat00 * val00 + e.mat01 * val10;
                    cuCpx w01 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[0][0], val01) + COMPLEX_MULTIPLY_REAL(e[0][1], val11),
                        COMPLEX_MULTIPLY_IMAG(e[0][0], val01) + COMPLEX_MULTIPLY_IMAG(e[0][1], val11)
                    ); // e.mat00 * val01 + e.mat01 * val11;
                    cuCpx w10 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[1][0], val00) + COMPLEX_MULTIPLY_REAL(e[1][1], val10),
                        COMPLEX_MULTIPLY_IMAG(e[1][0], val00) + COMPLEX_MULTIPLY_IMAG(e[1][1], val10)
                    ); // e.mat10 * val00 + e.mat11 * val10;
                    cuCpx w11 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[1][0], val01) + COMPLEX_MULTIPLY_REAL(e[1][1], val11),
                        COMPLEX_MULTIPLY_IMAG(e[1][0], val01) + COMPLEX_MULTIPLY_IMAG(e[1][1], val11)
                    ); // e.mat10 * val01 + e.mat11 * val11;

                    sum00 = make_cuComplex(
                        sum00.x + COMPLEX_MULTIPLY_REAL_ND(w00, e[0][0]) + COMPLEX_MULTIPLY_REAL_ND(w01, e[0][1]),
                        sum00.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e[0][0]) + COMPLEX_MULTIPLY_IMAG_ND(w01, e[0][1])
                    ); // sum00 + w00 * e.mat00 + w01 * e.mat01;
                    sum01 = make_cuComplex(
                        sum01.x + COMPLEX_MULTIPLY_REAL_ND(w00, e[1][0]) + COMPLEX_MULTIPLY_REAL_ND(w01, e[1][1]),
                        sum01.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e[1][0]) + COMPLEX_MULTIPLY_IMAG_ND(w01, e[1][1])
                    ); // sum01 + w00 * e.mat10 + w01 * e.mat11;
                    sum10 = make_cuComplex(
                        sum10.x + COMPLEX_MULTIPLY_REAL_ND(w10, e[0][0]) + COMPLEX_MULTIPLY_REAL_ND(w11, e[0][1]),
                        sum10.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e[0][0]) + COMPLEX_MULTIPLY_IMAG_ND(w11, e[0][1])
                    ); // sum10 + w10 * e.mat00 + w11 * e.mat01;
                    sum11 = make_cuComplex(
                        sum11.x + COMPLEX_MULTIPLY_REAL_ND(w10, e[1][0]) + COMPLEX_MULTIPLY_REAL_ND(w11, e[1][1]),
                        sum11.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e[1][0]) + COMPLEX_MULTIPLY_IMAG_ND(w11, e[1][1])
                    ); // sum11 + w10 * e.mat10 + w11 * e.mat11;
                }
                shm[s00] = sum00; shm[s01] = sum01; shm[s10] = sum10; shm[s11] = sum11;
            }
        }
        __syncthreads();
        if (deviceGates[i].err_len_control > 0) {
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 2);
            int qid = deviceGates[i].controlQubit == -3? deviceGates[i].encodeQubit: deviceGates[i].controlQubit;
            // if (blockIdx.x == 0 && threadIdx.x == 0) printf("qid = %d len %d\n", qid, deviceGates[i].err_len_control);
            qid *= 2;
            int numErrors = deviceGates[i].err_len_control;
            for (int j = threadIdx.x; j < m; j += blockSize) {
                int s00 = ((j >> qid) << (qid + 2)) | (j & ((1 << qid) - 1));
                int s01 = s00 | (1 << qid);
                int s10 = s00 | (1 << (qid + 1));
                int s11 = s01 | s10;
                // if (blockIdx.x == 0 && threadIdx.x == 0) printf("idx %d %d %d %d\n", s00, s01, s10, s11);
                cuCpx val00 = shm[s00], val01 = shm[s01], val10 = shm[s10], val11 = shm[s11];
                // if (blockIdx.x == 0 && threadIdx.x == 0) printf("load %lf %lf %lf %lf\n", val00.x, val01.x, val10.x, val11.x);
                cuCpx sum00 = make_cuComplex(0.0, 0.0), sum01 = make_cuComplex(0.0, 0.0), sum10 = make_cuComplex(0.0, 0.0), sum11 = make_cuComplex(0.0, 0.0);
                for (int k = 0; k < numErrors; k++) {
                    cuCpx (*e)[2] = reinterpret_cast<cuCpx(*)[2]>(deviceGates[i].errs_control[k]);
                    cuCpx w00 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[0][0], val00) + COMPLEX_MULTIPLY_REAL(e[0][1], val10),
                        COMPLEX_MULTIPLY_IMAG(e[0][0], val00) + COMPLEX_MULTIPLY_IMAG(e[0][1], val10)
                    ); // e.mat00 * val00 + e.mat01 * val10;
                    cuCpx w01 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[0][0], val01) + COMPLEX_MULTIPLY_REAL(e[0][1], val11),
                        COMPLEX_MULTIPLY_IMAG(e[0][0], val01) + COMPLEX_MULTIPLY_IMAG(e[0][1], val11)
                    ); // e.mat00 * val01 + e.mat01 * val11;
                    cuCpx w10 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[1][0], val00) + COMPLEX_MULTIPLY_REAL(e[1][1], val10),
                        COMPLEX_MULTIPLY_IMAG(e[1][0], val00) + COMPLEX_MULTIPLY_IMAG(e[1][1], val10)
                    ); // e.mat10 * val00 + e.mat11 * val10;
                    cuCpx w11 = make_cuComplex(
                        COMPLEX_MULTIPLY_REAL(e[1][0], val01) + COMPLEX_MULTIPLY_REAL(e[1][1], val11),
                        COMPLEX_MULTIPLY_IMAG(e[1][0], val01) + COMPLEX_MULTIPLY_IMAG(e[1][1], val11)
                    ); // e.mat10 * val01 + e.mat11 * val11;

                    sum00 = make_cuComplex(
                        sum00.x + COMPLEX_MULTIPLY_REAL_ND(w00, e[0][0]) + COMPLEX_MULTIPLY_REAL_ND(w01, e[0][1]),
                        sum00.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e[0][0]) + COMPLEX_MULTIPLY_IMAG_ND(w01, e[0][1])
                    ); // sum00 + w00 * e.mat00 + w01 * e.mat01;
                    sum01 = make_cuComplex(
                        sum01.x + COMPLEX_MULTIPLY_REAL_ND(w00, e[1][0]) + COMPLEX_MULTIPLY_REAL_ND(w01, e[1][1]),
                        sum01.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e[1][0]) + COMPLEX_MULTIPLY_IMAG_ND(w01, e[1][1])
                    ); // sum01 + w00 * e.mat10 + w01 * e.mat11;
                    sum10 = make_cuComplex(
                        sum10.x + COMPLEX_MULTIPLY_REAL_ND(w10, e[0][0]) + COMPLEX_MULTIPLY_REAL_ND(w11, e[0][1]),
                        sum10.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e[0][0]) + COMPLEX_MULTIPLY_IMAG_ND(w11, e[0][1])
                    ); // sum10 + w10 * e.mat00 + w11 * e.mat01;
                    sum11 = make_cuComplex(
                        sum11.x + COMPLEX_MULTIPLY_REAL_ND(w10, e[1][0]) + COMPLEX_MULTIPLY_REAL_ND(w11, e[1][1]),
                        sum11.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e[1][0]) + COMPLEX_MULTIPLY_IMAG_ND(w11, e[1][1])
                    ); // sum11 + w10 * e.mat10 + w11 * e.mat11;
                }
                // if (blockIdx.x == 0 && threadIdx.x == 0) printf("result %lf %lf %lf %lf\n", sum00.x, sum01.x, sum10.x, sum11.x);
                shm[s00] = sum00; shm[s01] = sum01; shm[s10] = sum10; shm[s11] = sum11;
            }
        }
        __syncthreads();
    }
}

__device__ void fetchDataDM(cuCpx* a, unsigned int* threadBias, unsigned int idx, unsigned int blockHot, unsigned int enumerate, int numLocalQubits) {
    if (threadIdx.x == 0) {
        int bid = blockIdx.x;
        unsigned int bias = 0;
        for (unsigned int bit = 1; bit < (1u << (numLocalQubits * 2)); bit <<= 1) {
            if (blockHot & bit) {
                if (bid & 1)
                    bias |= bit;
                bid >>= 1;
            }
        }
        blockBias = bias;
    }
    __syncthreads();
    unsigned int bias = blockBias | threadBias[threadIdx.x];
    int x;
    unsigned int y;
    for (x = ((1 << (LOCAL_QUBIT_SIZE * 2 - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        shm[x] = a[bias | y];
    }
}

__device__ void saveDataDM(cuCpx* a, unsigned int* threadBias, unsigned int enumerate) {
    unsigned int bias = blockBias | threadBias[threadIdx.x];
    int x;
    unsigned y;
    for (x = ((1 << (LOCAL_QUBIT_SIZE * 2 - THREAD_DEP)) - 1) << THREAD_DEP | threadIdx.x, y = enumerate;
        x >= 0;
        x -= (1 << THREAD_DEP), y = enumerate & (y - 1)) {
        
        a[bias | y] = shm[x];
    }
}

template <unsigned int blockSize>
__global__ void run_dm(cuCpx* a, unsigned int* threadBias, KernelGate* deviceGates, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate) {
    unsigned int idx = (unsigned int) blockIdx.x * blockSize + threadIdx.x;
    fetchDataDM(a, threadBias, idx, blockHot, enumerate, numLocalQubits);
    __syncthreads();
    doComputeDM<blockSize>(numGates, deviceGates);
    __syncthreads();
    saveDataDM(a, threadBias, enumerate);
}
#endif

__global__ void run_dm_control(cuCpx* a, int controlQubit, int targetQubit, const cuCpx v00, const cuCpx v01, const cuCpx v10, const cuCpx v11) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int smallQubit = min(controlQubit, targetQubit);
    int largeQubit = max(controlQubit, targetQubit);
    int lo = ((tid >> smallQubit) << (smallQubit + 1)) | (tid & ((1 << smallQubit) - 1));
    lo = ((lo >> largeQubit) << (largeQubit + 1)) | (lo & ((1 << largeQubit) - 1));
    lo |= 1 << controlQubit;
    int hi = lo | (1 << targetQubit);
    cuCpx val0 = a[lo];
    cuCpx val1 = a[hi];
    a[lo] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v00) + COMPLEX_MULTIPLY_REAL(val1, v01),
        COMPLEX_MULTIPLY_IMAG(val0, v00) + COMPLEX_MULTIPLY_IMAG(val1, v01)
    );
    a[hi] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v10) + COMPLEX_MULTIPLY_REAL(val1, v11),
        COMPLEX_MULTIPLY_IMAG(val0, v10) + COMPLEX_MULTIPLY_IMAG(val1, v11)
    );
}

__global__ void run_dm_single(cuCpx* a, int targetQubit, const cuCpx v00, const cuCpx v01, const cuCpx v10, const cuCpx v11) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lo = ((tid >> targetQubit) << (targetQubit + 1)) | ((tid & (1 << targetQubit) - 1));
    int hi = lo | (1 << targetQubit);
    cuCpx val0 = a[lo];
    cuCpx val1 = a[hi];
    a[lo] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v00) + COMPLEX_MULTIPLY_REAL(val1, v01),
        COMPLEX_MULTIPLY_IMAG(val0, v00) + COMPLEX_MULTIPLY_IMAG(val1, v01)
    );
    a[hi] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v10) + COMPLEX_MULTIPLY_REAL(val1, v11),
        COMPLEX_MULTIPLY_IMAG(val0, v10) + COMPLEX_MULTIPLY_IMAG(val1, v11)
    );
}

__global__ void run_dm_dual(cuCpx* a, int targetQubit1, int targetQubit2, const cuCpx v00, const cuCpx v01, const cuCpx v10, const cuCpx v11) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int smallQubit = min(targetQubit1, targetQubit2);
    int largeQubit = max(targetQubit1, targetQubit2);
    int s00 = ((tid >> smallQubit) << (smallQubit + 1)) | (tid & ((1 << smallQubit) - 1));
    s00 = ((s00 >> largeQubit) << (largeQubit + 1)) | (s00 & ((1 << largeQubit) - 1));
    int s01 = s00 | (1 << targetQubit2);
    int s10 = s00 | (1 << targetQubit1);
    int s11 = s01 | s10;

    cuCpx val0 = a[s00];
    cuCpx val1 = a[s11];
    a[s00] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v00) + COMPLEX_MULTIPLY_REAL(val1, v11),
        COMPLEX_MULTIPLY_IMAG(val0, v00) + COMPLEX_MULTIPLY_IMAG(val1, v11)
    );
    a[s11] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v11) + COMPLEX_MULTIPLY_REAL(val1, v00),
        COMPLEX_MULTIPLY_IMAG(val0, v11) + COMPLEX_MULTIPLY_IMAG(val1, v00)
    );

    val0 = a[s01];
    val1 = a[s10];
    a[s01] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v01) + COMPLEX_MULTIPLY_REAL(val1, v10),
        COMPLEX_MULTIPLY_IMAG(val0, v01) + COMPLEX_MULTIPLY_IMAG(val1, v10)
    );
    a[s10] = make_cuComplex(
        COMPLEX_MULTIPLY_REAL(val0, v10) + COMPLEX_MULTIPLY_REAL(val1, v01),
        COMPLEX_MULTIPLY_IMAG(val0, v10) + COMPLEX_MULTIPLY_IMAG(val1, v01)
    );
}

struct CudaError {
    CudaError() = default;
    GateType type;
    cuCpx mat00, mat01, mat10, mat11;
    CudaError& operator = (const Error& other) {
        type = other.type;
        mat00 = make_cuComplex(other.mat00.real(), other.mat00.imag());
        mat01 = make_cuComplex(other.mat01.real(), other.mat01.imag());
        mat10 = make_cuComplex(other.mat10.real(), other.mat10.imag());
        mat11 = make_cuComplex(other.mat11.real(), other.mat11.imag());
        return *this;
    }
};

__global__ void apply_error_single(cuCpx* a, int targetQubit, const CudaError* error, size_t numErrors) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int qid = targetQubit * 2;
    int s00 = ((tid >> qid) << (qid + 2)) | (tid & ((1 << qid) - 1));
    int s01 = s00 | (1 << qid);
    int s10 = s00 | (1 << (qid + 1));
    int s11 = s01 | s10;
    cuCpx val00 = a[s00], val01 = a[s01], val10 = a[s10], val11 = a[s11];
    cuCpx sum00 = make_cuComplex(0.0, 0.0), sum01 = make_cuComplex(0.0, 0.0), sum10 = make_cuComplex(0.0, 0.0), sum11 = make_cuComplex(0.0, 0.0);
    for (int i = 0; i < numErrors; i++) {
        const CudaError& e = error[i];
        cuCpx w00 = make_cuComplex(
            COMPLEX_MULTIPLY_REAL(e.mat00, val00) + COMPLEX_MULTIPLY_REAL(e.mat01, val10),
            COMPLEX_MULTIPLY_IMAG(e.mat00, val00) + COMPLEX_MULTIPLY_IMAG(e.mat01, val10)
        ); // e.mat00 * val00 + e.mat01 * val10;
        cuCpx w01 = make_cuComplex(
            COMPLEX_MULTIPLY_REAL(e.mat00, val01) + COMPLEX_MULTIPLY_REAL(e.mat01, val11),
            COMPLEX_MULTIPLY_IMAG(e.mat00, val01) + COMPLEX_MULTIPLY_IMAG(e.mat01, val11)
        ); // e.mat00 * val01 + e.mat01 * val11;
        cuCpx w10 = make_cuComplex(
            COMPLEX_MULTIPLY_REAL(e.mat10, val00) + COMPLEX_MULTIPLY_REAL(e.mat11, val10),
            COMPLEX_MULTIPLY_IMAG(e.mat10, val00) + COMPLEX_MULTIPLY_IMAG(e.mat11, val10)
        ); // e.mat10 * val00 + e.mat11 * val10;
        cuCpx w11 = make_cuComplex(
            COMPLEX_MULTIPLY_REAL(e.mat10, val01) + COMPLEX_MULTIPLY_REAL(e.mat11, val11),
            COMPLEX_MULTIPLY_IMAG(e.mat10, val01) + COMPLEX_MULTIPLY_IMAG(e.mat11, val11)
        ); // e.mat10 * val01 + e.mat11 * val11;

        sum00 = make_cuComplex(
            sum00.x + COMPLEX_MULTIPLY_REAL_ND(w00, e.mat00) + COMPLEX_MULTIPLY_REAL_ND(w01, e.mat01),
            sum00.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e.mat00) + COMPLEX_MULTIPLY_IMAG_ND(w01, e.mat01)
        ); // sum00 + w00 * e.mat00 + w01 * e.mat01;
        sum01 = make_cuComplex(
            sum01.x + COMPLEX_MULTIPLY_REAL_ND(w00, e.mat10) + COMPLEX_MULTIPLY_REAL_ND(w01, e.mat11),
            sum01.y + COMPLEX_MULTIPLY_IMAG_ND(w00, e.mat10) + COMPLEX_MULTIPLY_IMAG_ND(w01, e.mat11)
        ); // sum01 + w00 * e.mat10 + w01 * e.mat11;
        sum10 = make_cuComplex(
            sum10.x + COMPLEX_MULTIPLY_REAL_ND(w10, e.mat00) + COMPLEX_MULTIPLY_REAL_ND(w11, e.mat01),
            sum10.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e.mat00) + COMPLEX_MULTIPLY_IMAG_ND(w11, e.mat01)
        ); // sum10 + w10 * e.mat00 + w11 * e.mat01;
        sum11 = make_cuComplex(
            sum11.x + COMPLEX_MULTIPLY_REAL_ND(w10, e.mat10) + COMPLEX_MULTIPLY_REAL_ND(w11, e.mat11),
            sum11.y + COMPLEX_MULTIPLY_IMAG_ND(w10, e.mat10) + COMPLEX_MULTIPLY_IMAG_ND(w11, e.mat11)
        ); // sum11 + w10 * e.mat10 + w11 * e.mat11;
    }
    a[s00] = sum00; a[s01] = sum01; a[s10] = sum10; a[s11] = sum11;
}

}

void copyGatesToGlobal(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID) {
    checkCudaErrors(cudaMemcpyAsync(CudaDM::deviceGatePointers[gpuID], hostGates + gpuID * numGates, sizeof(KernelGate) * numGates, cudaMemcpyHostToDevice, stream));
}

void launchDMExecutor(int gridDim, cpx* deviceStateVec, unsigned int* threadBias, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate, cudaStream_t& stream, int gpuID) {
#if MODE == 2
    CudaDM::run_dm<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP, 0, stream>>>
        (reinterpret_cast<cuCpx*>(deviceStateVec), threadBias, CudaDM::deviceGatePointers[gpuID], numLocalQubits, numGates, blockHot, enumerate);
#else
    UNREACHABLE()
#endif
}

void apply_errors(cuCpx* deviceStateVec, CudaDM::CudaError* errors, idx_t nVec, int targetQubit, const std::vector<Error>& targetErrors) {
    CudaDM::CudaError hostErr[targetErrors.size()];
    for (int i = 0; i < (int) targetErrors.size(); i++)
        hostErr[i] = targetErrors[i];
    checkCudaErrors(cudaMemcpy(errors, hostErr, targetErrors.size() * sizeof(CudaDM::CudaError), cudaMemcpyHostToDevice));
    CudaDM::apply_error_single<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
        deviceStateVec, targetQubit, errors, targetErrors.size()
    );
}

void launchDMExecutorSerial(cpx* deviceStateVec_, int numLocalQubits, const std::vector<Gate>& gates) {
    checkCudaErrors(cudaDeviceSynchronize());
    cuCpx* deviceStateVec = reinterpret_cast<cuCpx*>(deviceStateVec_);
    // printf("numLocalQubits %d\n", numLocalQubits);
    idx_t nVec = idx_t(1) << (numLocalQubits * 2);
    CudaDM::CudaError* errors;
    checkCudaErrors(cudaMalloc(&errors, sizeof(CudaDM::CudaError) * MAX_ERROR_LEN));
    for (auto& gate: gates) {
        cuCpx mat[2][2];
        memcpy(mat, gate.mat, sizeof(cuCpx) * 4);
        if (gate.isControlGate()) {
            CudaDM::run_dm_control<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.controlQubit * 2, gate.targetQubit * 2,
                mat[0][0], mat[0][1], mat[1][0], mat[1][1]
            );
            CudaDM::run_dm_control<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.controlQubit * 2 + 1, gate.targetQubit * 2 + 1, 
                cuConj(mat[0][0]), cuConj(mat[0][1]), cuConj(mat[1][0]), cuConj(mat[1][1])
            );
            if (gate.controlErrors.size() > 0) {
                apply_errors(deviceStateVec, errors, nVec, gate.controlQubit, gate.controlErrors);
            }
            if (gate.targetErrors.size() > 0) {
                apply_errors(deviceStateVec, errors, nVec, gate.targetQubit, gate.targetErrors);
            }
        } else if (gate.isSingleGate()) {
            CudaDM::run_dm_single<<<nVec >> (THREAD_DEP + 1), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.targetQubit * 2,
                mat[0][0], mat[0][1], mat[1][0], mat[1][1]
            );
            CudaDM::run_dm_single<<<nVec >> (THREAD_DEP + 1), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.targetQubit * 2 + 1, 
                cuConj(mat[0][0]), cuConj(mat[0][1]), cuConj(mat[1][0]), cuConj(mat[1][1])
            );
            if (gate.targetErrors.size() > 0) {
                apply_errors(deviceStateVec, errors, nVec, gate.targetQubit, gate.targetErrors);
            }
        } else if (gate.isTwoQubitGate()) {
            CudaDM::run_dm_dual<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.encodeQubit * 2, gate.targetQubit * 2,
                mat[0][0], mat[0][1], mat[1][0], mat[1][1]
            );
            CudaDM::run_dm_dual<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.encodeQubit * 2 + 1, gate.targetQubit * 2 + 1,
                cuConj(mat[0][0]), cuConj(mat[0][1]), cuConj(mat[1][0]), cuConj(mat[1][1])
            );
            if (gate.controlErrors.size() > 0) {
                apply_errors(deviceStateVec, errors, nVec, gate.encodeQubit, gate.controlErrors);
            }
            if (gate.targetErrors.size() > 0) {
                apply_errors(deviceStateVec, errors, nVec, gate.targetQubit, gate.targetErrors);
            }
        } else {
            UNIMPLEMENTED();
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

void dmAllocGate(int localGPUs) {
    CudaDM::deviceGatePointers.resize(localGPUs);
    for (int i = 0; i < localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMalloc(&CudaDM::deviceGatePointers[i], sizeof(KernelGate) * MAX_GATE));
    }
}
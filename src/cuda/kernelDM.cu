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

static __shared__ cuCpx shm[1<<(LOCAL_QUBIT_SIZE * 2)];
static __shared__ idx_t blockBias;

__device__ __constant__ value_t recRoot2 = 0.70710678118654752440084436210485; // more elegant way?
std::vector<KernelGate*> deviceGatePointers;

#define COMPLEX_MULTIPLY_REAL(v0, v1) (v0.x * v1.x - v0.y * v1.y)
#define COMPLEX_MULTIPLY_IMAG(v0, v1) (v0.x * v1.y + v0.y * v1.x)

template <unsigned int blockSize>
__device__ void doComputeDM(int numGates, KernelGate* deviceGates) {
    for (int i = 0; i < numGates; i++) {
        int controlQubit = deviceGates[i].controlQubit;
        int targetQubit = deviceGates[i].targetQubit;
        if (deviceGates[i].controlQubit == -1) { // single qubit gate
            auto& gate = deviceGates[i];
            targetQubit *= 2;
            int smallQubit = targetQubit;
            int m = 1 << (LOCAL_QUBIT_SIZE * 2 - 1);
            int maskTarget = (1 << smallQubit) - 1;
            for (int j = threadIdx.x; j < m; j += blockSize) {
                int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskTarget);
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
            smallQubit ++;
            maskTarget = (1 << smallQubit) - 1;
            for (int j = threadIdx.x; j < m; j += blockSize) {
                int s0 = ((j >> smallQubit) << (smallQubit + 1)) | (j & maskTarget);
                int s1 = s0 | (1 << smallQubit);
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
    int s00 = ((tid >> smallQubit) << (smallQubit + 1)) | (tid &((1 << smallQubit) - 1));
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

}

void copyGatesToGlobal(KernelGate* hostGates, int numGates, cudaStream_t& stream, int gpuID) {
    checkCudaErrors(cudaMemcpyAsync(CudaDM::deviceGatePointers[gpuID], hostGates + gpuID * numGates, sizeof(KernelGate) * numGates, cudaMemcpyHostToDevice, stream));
}

void launchDMExecutor(int gridDim, cpx* deviceStateVec, unsigned int* threadBias, int numLocalQubits, int numGates, unsigned int blockHot, unsigned int enumerate, cudaStream_t& stream, int gpuID) {
    CudaDM::run_dm<1<<THREAD_DEP><<<gridDim, 1<<THREAD_DEP, 0, stream>>>
        (reinterpret_cast<cuCpx*>(deviceStateVec), threadBias, CudaDM::deviceGatePointers[gpuID], numLocalQubits, numGates, blockHot, enumerate);
}

void launchDMExecutorSerial(cpx* deviceStateVec_, int numLocalQubits, const std::vector<Gate>& gates) {
    checkCudaErrors(cudaDeviceSynchronize());
    cuCpx* deviceStateVec = reinterpret_cast<cuCpx*>(deviceStateVec_);
    // printf("numLocalQubits %d\n", numLocalQubits);
    idx_t nVec = idx_t(1) << (numLocalQubits * 2);
    for (auto& gate: gates) {
        printf("run %lld %d %d\n", gate.encodeQubit, gate.controlQubit, gate.targetQubit);
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
        } else if (gate.isSingleGate()) {
            CudaDM::run_dm_single<<<nVec >> (THREAD_DEP + 1), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.targetQubit * 2,
                mat[0][0], mat[0][1], mat[1][0], mat[1][1]
            );
            CudaDM::run_dm_single<<<nVec >> (THREAD_DEP + 1), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.targetQubit * 2 + 1, 
                cuConj(mat[0][0]), cuConj(mat[0][1]), cuConj(mat[1][0]), cuConj(mat[1][1])
            );
        } else if (gate.isTwoQubitGate()) {
            CudaDM::run_dm_control<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.encodeQubit * 2, gate.targetQubit * 2,
                mat[0][0], mat[0][1], mat[1][0], mat[1][1]
            );
            CudaDM::run_dm_control<<<nVec >> (THREAD_DEP + 2), 1 << THREAD_DEP>>>(
                deviceStateVec, gate.encodeQubit * 2 + 1, gate.targetQubit * 2 + 1,
                cuConj(mat[0][0]), cuConj(mat[0][1]), cuConj(mat[1][0]), cuConj(mat[1][1])
            );
        }
        checkCudaErrors(cudaDeviceSynchronize());
    }
    printf("run end\n");
}

void dmAllocGate(int localGPUs) {
    CudaDM::deviceGatePointers.resize(localGPUs);
    for (int i = 0; i < localGPUs; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMalloc(&CudaDM::deviceGatePointers[i], sizeof(KernelGate) * MAX_GATE));
    }
}